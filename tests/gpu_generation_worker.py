"""在独立进程中运行一次确定性的 GPU 正确性测试场景。"""

import argparse
import atexit
import json

import torch

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
        choices=("eager", "cuda_graph", "prefix_miss", "prefix_hit"),
        required=True,
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


def main() -> None:
    args = parse_args()
    # 只修改当前测试子进程中的 Sampler；正式 nano-vLLM 代码仍使用原有随机采样。
    Sampler.forward = deterministic_forward

    is_prefix_scenario = args.mode.startswith("prefix_")
    llm = LLM(
        args.model,
        # Prefix Cache 测试固定使用 Eager，避免把 CUDA Graph 变成第二个变量。
        enforce_eager=args.mode != "cuda_graph",
        max_model_len=512 if is_prefix_scenario else 256,
        max_num_batched_tokens=512 if is_prefix_scenario else 256,
        # 当前 CUDA Graph 固定提供 1/2/4/8 这几个小 batch bucket。
        max_num_seqs=8,
        # Prefix 测试通过指标证明 Hit 组确实走了缓存，而不是两边都完整 Prefill。
        enable_stats=is_prefix_scenario,
    )
    # LLMEngine 默认注册 atexit；worker 选择在 finally 中主动清理，因此先取消重复调用。
    atexit.unregister(llm.exit)
    try:
        torch.manual_seed(0)
        payload = (
            run_prefix_cache_scenario(llm, args.mode)
            if is_prefix_scenario
            else run_eager_graph_scenario(llm, args.mode)
        )
    finally:
        llm.exit()

    print(RESULT_PREFIX + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
