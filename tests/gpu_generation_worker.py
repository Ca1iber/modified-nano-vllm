"""在独立进程中运行一次确定性的 GPU 正确性测试场景。"""

import argparse
import atexit
import json

import torch

from bench import build_sampling_params, run_setup_requests, run_target_requests
from bench_workloads import build_workload_spec
from nanovllm import LLM, SamplingParams
from nanovllm.layers.sampler import Sampler

'''
 如果在同一个 pytest 进程里直接创建多个 LLM：

  eager_llm = LLM(...)
  graph_llm = LLM(...)

  会遇到几个问题：

  - 两个模型可能同时占用 GPU 显存；
  - NCCL process group 属于全局状态，重复初始化可能冲突；
  - CUDA Graph 会留下自己的内存池；
  - 第一个引擎的 KV Cache 可能影响第二个引擎；
  - Prefix Cache Miss 测试需要真正干净的缓存环境。

  所以 pytest 采用：

  subprocess.run(...)

  每次启动一个独立 Python 子进程。子进程结束后，模型、NCCL、KV Cache 和 CUDA Graph 状态全部一起销毁。
'''

RESULT_PREFIX = "NANOVLLM_GPU_RESULT="
EAGER_GRAPH_PROMPTS = [
    [10, 11, 12, 13],
    [20, 21, 22, 23, 24],
    [30, 31, 32, 33, 34, 35],
]
EAGER_GRAPH_OUTPUT_LENGTHS = [8, 6, 4]

# nano-vLLM 当前一个 KV Block 固定容纳 256 个 token。两个 Prompt 只共享第一个
# 完整 Block，尾部故意不同，用来验证“复用公共前缀 + 计算新尾部”的真实路径。
SHARED_PREFIX = list(range(1000, 1256))
PREFIX_PRIMER_PROMPT = SHARED_PREFIX + [2000, 2001, 2002, 2003]
PREFIX_TARGET_PROMPT = SHARED_PREFIX + [3000, 3001, 3002, 3003]
PREFIX_TARGET_OUTPUT_LENGTH = 4


def deterministic_forward(
    self: Sampler,
    logits: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    """用确定性 argmax 隔离 Eager/CUDA Graph 差异，不让随机采样污染结果。"""
    del self, temperatures
    return logits.argmax(dim=-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode",
        choices=(
            "eager",
            "cuda_graph",
            "prefix_miss",
            "prefix_hit",
            "shared_prefix_workload",
            "kv_pressure_relaxed",
            "kv_pressure_tight",
            "dynamic_arrival_workload",
            "official_workload",
        ),
        required=True,
    )
    parser.add_argument(
        "--workload",
        help="mode=official_workload 时要运行的 P0.4 workload 名称。",
    )
    return parser.parse_args()


def run_eager_graph_scenario(llm: LLM, mode: str) -> dict:
    """生成 8/6/4 个 token，供 Eager 与 CUDA Graph 两个进程对照。"""
    sampling_params = [
        SamplingParams(ignore_eos=True, max_tokens=output_length)
        for output_length in EAGER_GRAPH_OUTPUT_LENGTHS
    ]
    outputs = llm.generate(
        EAGER_GRAPH_PROMPTS,
        sampling_params,
        use_tqdm=False,
    )
    return {
        "mode": mode,
        "expected_output_lengths": EAGER_GRAPH_OUTPUT_LENGTHS,
        "token_ids": [output["token_ids"] for output in outputs],
    }


def run_prefix_cache_scenario(llm: LLM, mode: str) -> dict:
    """在干净缓存或预热过公共前缀的缓存上生成同一个目标请求。"""
    if mode == "prefix_hit":
        # Primer 完成后会释放 Block 引用，但完整公共前缀的 hash 和 GPU K/V 会保留，
        # 随后的目标请求应复用这一个 256-token Block。
        llm.generate(
            [PREFIX_PRIMER_PROMPT],
            SamplingParams(ignore_eos=True, max_tokens=1),
            use_tqdm=False,
        )

    outputs = llm.generate(
        [PREFIX_TARGET_PROMPT],
        SamplingParams(
            ignore_eos=True,
            max_tokens=PREFIX_TARGET_OUTPUT_LENGTH,
        ),
        use_tqdm=False,
    )
    metrics_summary = llm.get_metrics_summary()
    assert metrics_summary is not None
    return {
        "mode": mode,
        "expected_output_lengths": [PREFIX_TARGET_OUTPUT_LENGTH],
        "token_ids": [output["token_ids"] for output in outputs],
        "prefix_hit_blocks": metrics_summary.total_prefix_hit_blocks,
        "prefix_hit_tokens": metrics_summary.total_prefix_hit_tokens,
    }


def run_shared_prefix_workload(llm: LLM) -> dict:
    """真实运行 P0.4b 的 1 Primer + 16 Target，并返回正式 Target 指标。"""
    workload = build_workload_spec("shared_prefix_high_hit", seed=0)
    run_setup_requests(llm, workload, SamplingParams)
    target_params = build_sampling_params(workload.requests, SamplingParams)
    outputs = run_target_requests(llm, workload, target_params)
    metrics_summary = llm.get_metrics_summary()
    assert metrics_summary is not None
    return {
        "mode": "shared_prefix_workload",
        "request_count": metrics_summary.request_count,
        "completed_request_count": metrics_summary.completed_request_count,
        "output_lengths": [len(output["token_ids"]) for output in outputs],
        "total_output_tokens": sum(len(output["token_ids"]) for output in outputs),
        "prefix_hit_blocks": metrics_summary.total_prefix_hit_blocks,
        "prefix_hit_tokens": metrics_summary.total_prefix_hit_tokens,
    }


def run_kv_pressure_scenario(llm: LLM, mode: str) -> dict:
    """运行相同的两个请求，返回宽松/紧张 KV 容量下的输出和抢占指标。"""
    workload = build_workload_spec("kv_pressure_preemption", seed=0)
    target_params = build_sampling_params(workload.requests, SamplingParams)
    outputs = run_target_requests(llm, workload, target_params)
    metrics = llm.get_metrics_summary()
    assert metrics is not None
    block_manager = llm.scheduler.block_manager
    return {
        "mode": mode,
        "token_ids": [output["token_ids"] for output in outputs],
        "request_count": metrics.request_count,
        "completed_request_count": metrics.completed_request_count,
        "preemptions": metrics.total_preemptions,
        "preempted_cached_tokens": metrics.total_preempted_cached_tokens,
        "released_block_references": metrics.total_released_block_references,
        "freed_physical_blocks": metrics.total_freed_physical_blocks,
        "recompute_steps": metrics.total_recompute_steps,
        "recomputed_tokens": metrics.total_recomputed_tokens,
        "peak_used_kv_blocks": metrics.peak_used_kv_blocks,
        "used_kv_blocks_after_finish": len(block_manager.used_block_ids),
        "free_kv_blocks_after_finish": len(block_manager.free_block_ids),
    }


def run_dynamic_arrival_workload(llm: LLM) -> dict:
    """真实运行 P0.4d，并返回到达前后 Step 顺序、输出和资源状态。"""
    workload = build_workload_spec("decode_then_long_prefill", seed=0)
    target_params = build_sampling_params(workload.requests, SamplingParams)
    result = run_target_requests(llm, workload, target_params)
    metrics = llm.get_metrics_summary()
    assert metrics is not None
    steps = llm.get_step_metrics()
    request_metrics = llm.get_request_metrics()
    arrival_time = result.arrival_step_times[9]
    old_tokens_before_arrival = [
        sum(
            output_time <= arrival_time
            for output_time in request_metrics[seq_id].output_token_times
        )
        for seq_id in result.request_seq_ids[:4]
    ]
    block_manager = llm.scheduler.block_manager
    return {
        "mode": "dynamic_arrival_workload",
        "request_count": metrics.request_count,
        "completed_request_count": metrics.completed_request_count,
        "output_lengths": [len(output["token_ids"]) for output in result.outputs],
        "request_seq_ids": result.request_seq_ids,
        "old_tokens_before_arrival": old_tokens_before_arrival,
        "step_num_prefill_tokens": [step.num_prefill_tokens for step in steps],
        "step_num_decode_tokens": [step.num_decode_tokens for step in steps],
        "step_num_seqs": [step.num_seqs for step in steps],
        "step_num_tokens": [step.num_tokens for step in steps],
        "preemptions": metrics.total_preemptions,
        "recomputed_tokens": metrics.total_recomputed_tokens,
        "peak_used_kv_blocks": metrics.peak_used_kv_blocks,
        "used_kv_blocks_after_finish": len(block_manager.used_block_ids),
        "free_kv_blocks_after_finish": len(block_manager.free_block_ids),
    }


def run_official_workload(llm: LLM, workload_name: str) -> dict:
    """运行任意 P0.4 正式 workload，返回统一的 GPU 正确性字段。"""
    workload = build_workload_spec(workload_name, seed=0)
    run_setup_requests(llm, workload, SamplingParams)
    target_params = build_sampling_params(workload.requests, SamplingParams)
    result = run_target_requests(llm, workload, target_params)
    metrics = llm.get_metrics_summary()
    assert metrics is not None
    steps = llm.get_step_metrics()
    block_manager = llm.scheduler.block_manager
    payload = {
        "mode": "official_workload",
        "workload": workload_name,
        "request_count": metrics.request_count,
        "completed_request_count": metrics.completed_request_count,
        "expected_output_lengths": workload.output_lengths,
        "output_lengths": [len(output["token_ids"]) for output in result.outputs],
        "prefill_steps": metrics.prefill_step_count,
        "decode_steps": metrics.decode_step_count,
        "mixed_steps": metrics.mixed_step_count,
        "preemptions": metrics.total_preemptions,
        "recomputed_tokens": metrics.total_recomputed_tokens,
        "prefix_hit_blocks": metrics.total_prefix_hit_blocks,
        "prefix_hit_tokens": metrics.total_prefix_hit_tokens,
        "step_num_prefill_tokens": [step.num_prefill_tokens for step in steps],
        "step_num_decode_tokens": [step.num_decode_tokens for step in steps],
        "step_num_seqs": [step.num_seqs for step in steps],
        "step_num_tokens": [step.num_tokens for step in steps],
        "peak_used_kv_blocks": metrics.peak_used_kv_blocks,
        "used_kv_blocks_after_finish": len(block_manager.used_block_ids),
        "free_kv_blocks_after_finish": len(block_manager.free_block_ids),
    }
    if workload_name == "decode_then_long_prefill":
        request_metrics = llm.get_request_metrics()
        arrival_time = result.arrival_step_times[9]
        payload["old_tokens_before_arrival"] = [
            sum(
                output_time <= arrival_time
                for output_time in request_metrics[seq_id].output_token_times
            )
            for seq_id in result.request_seq_ids[:4]
        ]
    return payload


def main() -> None:
    args = parse_args()
    # 只修改当前测试子进程中的 Sampler；正式 nano-vLLM 代码仍使用原有随机采样。
    Sampler.forward = deterministic_forward

    is_prefix_scenario = (
        args.mode.startswith("prefix_")
        or args.mode == "shared_prefix_workload"
    )
    is_shared_prefix_workload = args.mode == "shared_prefix_workload"
    is_kv_pressure_scenario = args.mode.startswith("kv_pressure_")
    is_dynamic_arrival_workload = args.mode == "dynamic_arrival_workload"
    is_official_workload = args.mode == "official_workload"
    if is_official_workload:
        if args.workload is None:
            raise ValueError("official_workload mode 必须提供 --workload")
        official_spec = build_workload_spec(args.workload, seed=0)
        num_kvcache_blocks = official_spec.num_kvcache_blocks
    elif is_dynamic_arrival_workload:
        num_kvcache_blocks = 9
    elif is_kv_pressure_scenario:
        num_kvcache_blocks = 2 if args.mode == "kv_pressure_tight" else 4
    else:
        num_kvcache_blocks = -1
    if is_official_workload:
        max_model_len = official_spec.max_model_len
        max_num_batched_tokens = official_spec.max_num_batched_tokens
        max_num_seqs = official_spec.max_num_seqs
    elif is_dynamic_arrival_workload:
        max_model_len, max_num_batched_tokens, max_num_seqs = 2048, 512, 8
    elif is_shared_prefix_workload:
        max_model_len, max_num_batched_tokens, max_num_seqs = 1024, 512, 16
    elif is_kv_pressure_scenario:
        max_model_len, max_num_batched_tokens, max_num_seqs = 512, 256, 2
    elif is_prefix_scenario:
        max_model_len, max_num_batched_tokens, max_num_seqs = 512, 512, 8
    else:
        max_model_len, max_num_batched_tokens, max_num_seqs = 256, 256, 8
    llm = LLM(
        args.model,
        # 统一 workload 验收走与正式 benchmark 相同的 CUDA Graph；原有 Prefix、
        # KV 抢占和动态专项仍固定 Eager，继续隔离各自正在验证的状态变量。
        enforce_eager=args.mode not in ("cuda_graph", "official_workload"),
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        # 当前 CUDA Graph 固定提供 1/2/4/8 这几个小 batch bucket。
        max_num_seqs=max_num_seqs,
        # Prefix 测试通过指标证明 Hit 组确实走了缓存，而不是两边都完整 Prefill。
        enable_stats=(
            is_prefix_scenario
            or is_kv_pressure_scenario
            or is_dynamic_arrival_workload
            or is_official_workload
        ),
        num_kvcache_blocks=num_kvcache_blocks,
    )
    # LLMEngine 默认注册 atexit；worker 选择在 finally 中主动清理，因此先取消重复调用。
    atexit.unregister(llm.exit)
    try:
        torch.manual_seed(0)
        if is_official_workload:
            payload = run_official_workload(llm, args.workload)
        elif is_dynamic_arrival_workload:
            payload = run_dynamic_arrival_workload(llm)
        elif is_shared_prefix_workload:
            payload = run_shared_prefix_workload(llm)
        elif is_kv_pressure_scenario:
            payload = run_kv_pressure_scenario(llm, args.mode)
        elif is_prefix_scenario:
            payload = run_prefix_cache_scenario(llm, args.mode)
        else:
            payload = run_eager_graph_scenario(llm, args.mode)
    finally:
        llm.exit()

    print(RESULT_PREFIX + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
