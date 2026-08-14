"""用相同的 16 个 Target 对比 Prefix Cache Miss 与 Hit 的端到端性能。"""

import argparse
import atexit
import hashlib
import json
import os
from pathlib import Path
import subprocess
from statistics import median
import sys
import time

from bench import build_sampling_params, run_setup_requests, run_target_requests
from bench_workloads import WorkloadSpec, build_workload_spec


DEFAULT_MODEL_PATH = "~/huggingface/Qwen3-0.6B/"
WORKLOAD_NAME = "shared_prefix_high_hit"
RESULT_PREFIX = "NANOVLLM_PREFIX_AB_RESULT="
# 16 条 544-token Target 可以在同一个 Prefill step 中同时分配和执行。这样 Miss
# 组不会让第一条 Target 先完成并意外充当后续请求的 Primer。
AB_TOKEN_BUDGET = 16 * 544


def positive_int(value: str) -> int:
    """把命令行参数解析为正整数。"""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return parsed_value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "用两个独立引擎对比 16 个相同 Target 在 Prefix Cache Miss/Hit 下的性能。"
        )
    )
    parser.add_argument(
        "--repeat",
        type=positive_int,
        default=5,
        help="Miss 和 Hit 各自重复多少轮正式测量（默认：5）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="第一轮请求 seed；后续每轮依次加 1（默认：0）。",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"模型目录（默认：{DEFAULT_MODEL_PATH}）。",
    )
    # 父进程通过同一脚本启动两个内部 worker；不向普通用户展示这个实现参数。
    parser.add_argument(
        "--worker-mode",
        choices=("miss", "hit"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def target_digest(workload: WorkloadSpec) -> str:
    """生成正式 Target 输入摘要，用来证明 Miss/Hit 使用完全相同的 token。"""
    payload = json.dumps(workload.prompts, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_worker_command(
    script_path: Path,
    model_path: str,
    mode: str,
    repeat: int,
    seed: int,
) -> list[str]:
    """构造独立 Miss 或 Hit 引擎的子进程命令。"""
    return [
        sys.executable,
        str(script_path),
        "--model",
        model_path,
        "--worker-mode",
        mode,
        "--repeat",
        str(repeat),
        "--seed",
        str(seed),
    ]


def run_worker_process(
    script_path: Path,
    model_path: str,
    mode: str,
    repeat: int,
    seed: int,
) -> dict:
    """运行一个干净引擎，并从其 stdout 读取结构化 benchmark 结果。"""
    completed = subprocess.run(
        build_worker_command(script_path, model_path, mode, repeat, seed),
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        timeout=1200,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Prefix Cache {mode} worker 失败（exit={completed.returncode}）\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    raise RuntimeError(
        f"Prefix Cache {mode} worker 没有输出结果标记 {RESULT_PREFIX}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def run_worker(args: argparse.Namespace) -> dict:
    """在一个独立 GPU 引擎中完成 Miss 或 Hit 的多轮正式测量。"""
    import torch

    from nanovllm import LLM, SamplingParams

    model_path = os.path.expanduser(args.model)
    base_workload = build_workload_spec(WORKLOAD_NAME, args.seed)
    llm = LLM(
        model_path,
        enforce_eager=False,
        max_model_len=base_workload.max_model_len,
        # A/B 两组都使用同一大 budget，保证 16 个 Target 同轮 Prefill。
        max_num_batched_tokens=AB_TOKEN_BUDGET,
        max_num_seqs=base_workload.max_num_seqs,
        enable_stats=True,
    )
    # worker 在 finally 中主动释放 GPU/NCCL，避免 atexit 对同一引擎重复清理。
    atexit.unregister(llm.exit)
    runs = []
    try:
        # 两个 worker 都执行相同的非正式预热，排除首次运行固定开销。
        torch.manual_seed(args.seed)
        llm.generate(
            ["Prefix cache A/B warmup"],
            SamplingParams(ignore_eos=True, max_tokens=32),
            use_tqdm=False,
        )

        for run_index in range(args.repeat):
            workload_seed = args.seed + run_index
            workload = build_workload_spec(WORKLOAD_NAME, workload_seed)
            if args.worker_mode == "hit":
                # Primer 只为当前 seed 建立两个公共 Prefix Blocks，不计入正式时间。
                run_setup_requests(llm, workload, SamplingParams)

            # Primer 会消耗一次采样；在正式 Target 前重新设 seed，让两组采样起点一致。
            torch.manual_seed(workload_seed)
            sampling_params = build_sampling_params(workload.requests, SamplingParams)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            outputs = run_target_requests(llm, workload, sampling_params)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_time

            metrics = llm.get_metrics_summary()
            assert metrics is not None
            steps = llm.get_step_metrics()
            total_output_tokens = sum(len(output["token_ids"]) for output in outputs)
            runs.append(
                {
                    "seed": workload_seed,
                    "target_digest": target_digest(workload),
                    "request_count": metrics.request_count,
                    "completed_request_count": metrics.completed_request_count,
                    "total_output_tokens": total_output_tokens,
                    "elapsed_seconds": elapsed,
                    "throughput": total_output_tokens / elapsed,
                    "prefill_steps": metrics.prefill_step_count,
                    "prefill_computed_tokens": sum(
                        step.num_tokens for step in steps if step.is_prefill
                    ),
                    "prefix_hit_blocks": metrics.total_prefix_hit_blocks,
                    "prefix_hit_tokens": metrics.total_prefix_hit_tokens,
                    "ttft_p50_ms": metrics.ttft.p50 * 1000,
                    "ttft_p99_ms": metrics.ttft.p99 * 1000,
                    "e2e_p50_ms": metrics.e2e_latency.p50 * 1000,
                }
            )
    finally:
        llm.exit()
    return {"mode": args.worker_mode, "runs": runs}


def validate_comparison(miss_result: dict, hit_result: dict, repeat: int, seed: int) -> None:
    """严格验证 A/B 公平性和缓存工作量；不对易波动的实际耗时下硬断言。"""
    if miss_result["mode"] != "miss" or hit_result["mode"] != "hit":
        raise ValueError("Prefix Cache A/B worker 模式不匹配")
    miss_runs = miss_result["runs"]
    hit_runs = hit_result["runs"]
    if len(miss_runs) != repeat or len(hit_runs) != repeat:
        raise ValueError("Prefix Cache A/B worker 返回的轮数不正确")

    expected_seeds = list(range(seed, seed + repeat))
    if [run["seed"] for run in miss_runs] != expected_seeds:
        raise ValueError("Miss 组 seed 顺序不正确")
    if [run["seed"] for run in hit_runs] != expected_seeds:
        raise ValueError("Hit 组 seed 顺序不正确")

    for miss_run, hit_run in zip(miss_runs, hit_runs):
        if miss_run["target_digest"] != hit_run["target_digest"]:
            raise ValueError("Miss/Hit 没有使用完全相同的 Target token")
        for run in (miss_run, hit_run):
            if run["request_count"] != 16 or run["completed_request_count"] != 16:
                raise ValueError("正式请求没有全部完成")
            if run["total_output_tokens"] != 512:
                raise ValueError("正式输出 token 数不是预期的 512")
            if run["prefill_steps"] != 1:
                raise ValueError("A/B 正式 Target 没有在一个 Prefill step 中完成")
        if miss_run["prefix_hit_blocks"] != 0 or miss_run["prefix_hit_tokens"] != 0:
            raise ValueError("Miss 组意外命中了 Prefix Cache")
        if miss_run["prefill_computed_tokens"] != 8704:
            raise ValueError("Miss 组 Prefill 实算 token 数不是预期的 8704")
        if hit_run["prefix_hit_blocks"] != 32 or hit_run["prefix_hit_tokens"] != 8192:
            raise ValueError("Hit 组没有命中预期的 32 Blocks/8192 tokens")
        if hit_run["prefill_computed_tokens"] != 512:
            raise ValueError("Hit 组 Prefill 实算 token 数不是预期的 512")


def summarize_comparison(miss_result: dict, hit_result: dict) -> dict[str, float]:
    """使用多轮中位数计算耗时、吞吐、延迟和加速比。"""
    miss_runs = miss_result["runs"]
    hit_runs = hit_result["runs"]
    miss_time = median(run["elapsed_seconds"] for run in miss_runs)
    hit_time = median(run["elapsed_seconds"] for run in hit_runs)
    return {
        "miss_time": miss_time,
        "hit_time": hit_time,
        "time_speedup": miss_time / hit_time,
        "miss_throughput": median(run["throughput"] for run in miss_runs),
        "hit_throughput": median(run["throughput"] for run in hit_runs),
        "miss_ttft_p50_ms": median(run["ttft_p50_ms"] for run in miss_runs),
        "hit_ttft_p50_ms": median(run["ttft_p50_ms"] for run in hit_runs),
        "miss_ttft_p99_ms": median(run["ttft_p99_ms"] for run in miss_runs),
        "hit_ttft_p99_ms": median(run["ttft_p99_ms"] for run in hit_runs),
        "miss_e2e_p50_ms": median(run["e2e_p50_ms"] for run in miss_runs),
        "hit_e2e_p50_ms": median(run["e2e_p50_ms"] for run in hit_runs),
    }


def print_worker_runs(result: dict) -> None:
    """打印单组每轮原始结果，避免只保留最好的一次。"""
    label = result["mode"].upper()
    for index, run in enumerate(result["runs"], start=1):
        print(
            f"{label} Run {index}: seed={run['seed']}, "
            f"time={run['elapsed_seconds']:.3f}s, "
            f"throughput={run['throughput']:.2f}tok/s, "
            f"prefill_computed={run['prefill_computed_tokens']}tok, "
            f"prefix_hit={run['prefix_hit_tokens']}tok"
        )


def run_parent(args: argparse.Namespace) -> None:
    """依次启动干净的 Miss/Hit 引擎并输出 A/B 对照。"""
    script_path = Path(__file__).resolve()
    model_path = os.path.expanduser(args.model)
    print(
        f"Prefix Cache A/B Benchmark: workload={WORKLOAD_NAME}, "
        f"repeats={args.repeat}, base_seed={args.seed}"
    )
    print(
        "Formal targets: 16 requests, 544-token prompt, 32-token output; "
        "Primer is excluded from Hit timing"
    )
    miss_result = run_worker_process(
        script_path, model_path, "miss", args.repeat, args.seed
    )
    hit_result = run_worker_process(
        script_path, model_path, "hit", args.repeat, args.seed
    )
    validate_comparison(miss_result, hit_result, args.repeat, args.seed)
    summary = summarize_comparison(miss_result, hit_result)
    print_worker_runs(miss_result)
    print_worker_runs(hit_result)
    print("Prefix Cache A/B Summary (median)")
    print(
        f"Miss: time={summary['miss_time']:.3f}s, "
        f"throughput={summary['miss_throughput']:.2f}tok/s, "
        f"TTFT P50/P99={summary['miss_ttft_p50_ms']:.2f}/"
        f"{summary['miss_ttft_p99_ms']:.2f}ms, "
        "prefill_computed=8704tok, prefix_hit=0tok"
    )
    print(
        f"Hit:  time={summary['hit_time']:.3f}s, "
        f"throughput={summary['hit_throughput']:.2f}tok/s, "
        f"TTFT P50/P99={summary['hit_ttft_p50_ms']:.2f}/"
        f"{summary['hit_ttft_p99_ms']:.2f}ms, "
        "prefill_computed=512tok, prefix_hit=8192tok"
    )
    print(
        f"Speedup: {summary['time_speedup']:.2f}x; "
        "Prefill compute reduction: 94.12%"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker_mode is not None:
        print(RESULT_PREFIX + json.dumps(run_worker(args)), flush=True)
        return
    run_parent(args)


if __name__ == "__main__":
    main()
