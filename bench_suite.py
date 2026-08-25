"""独立进程运行 P0.4 六张正式 workload，并保存统一的多轮 GPU 基线。"""

import argparse
import atexit
import json
import os
from pathlib import Path
import platform
import subprocess
from statistics import median
import sys
import time

from bench import (
    build_sampling_params,
    format_workload,
    run_workload_round,
    summarize_dynamic_arrival,
)
from bench_workloads import OFFICIAL_WORKLOAD_NAMES, build_workload_spec
from nanovllm.config import SCHEDULER_POLICIES


DEFAULT_MODEL_PATH = "~/huggingface/Qwen3-0.6B/"
DEFAULT_REPORT_PATH = "benchmarks/P0_4_BASELINE.md"
RESULT_PREFIX = "NANOVLLM_SUITE_RESULT="
# 预热与正式测量必须使用两个互不重叠的正整数 seed 域。不能用负数隔离，因为
# Python random.Random(-1) 与 random.Random(1) 会生成相同序列，进而让正式轮次
# 意外命中预热留下的 Prefix Cache。
WARMUP_SEED_OFFSET = 1_000_000_000


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("不能小于 0")
    return parsed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 P0.4 六场景 GPU benchmark suite 并生成 Markdown 基线报告。"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--warmup", type=non_negative_int, default=1)
    parser.add_argument("--repeat", type=positive_int, default=3)
    parser.add_argument("--seed", type=non_negative_int, default=0)
    parser.add_argument("--report", default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--scheduler-policy",
        choices=SCHEDULER_POLICIES,
        default="prefill_first",
    )
    parser.add_argument("--time-sliced-decode-steps", type=positive_int, default=4)
    parser.add_argument(
        "--workload",
        choices=OFFICIAL_WORKLOAD_NAMES,
        action="append",
        dest="workloads",
        help="只运行指定正式 workload；可重复传入。默认运行全部六个。",
    )
    parser.add_argument(
        "--worker-workload",
        choices=OFFICIAL_WORKLOAD_NAMES,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args(argv)


def _milliseconds(value: float | None) -> float | None:
    return None if value is None else value * 1000


def metrics_to_record(metrics) -> dict:
    """把 EngineMetricsSummary 转成可跨进程保存的 JSON 字段。"""
    return {
        "request_count": metrics.request_count,
        "completed_request_count": metrics.completed_request_count,
        "step_count": metrics.step_count,
        "prefill_steps": metrics.prefill_step_count,
        "decode_steps": metrics.decode_step_count,
        "queue_wait_p50_ms": _milliseconds(metrics.queue_wait.p50),
        "queue_wait_p99_ms": _milliseconds(metrics.queue_wait.p99),
        "ttft_p50_ms": _milliseconds(metrics.ttft.p50),
        "ttft_p99_ms": _milliseconds(metrics.ttft.p99),
        "itl_p50_ms": _milliseconds(metrics.itl.p50),
        "itl_p99_ms": _milliseconds(metrics.itl.p99),
        "itl_max_ms": _milliseconds(metrics.itl.maximum),
        "e2e_p50_ms": _milliseconds(metrics.e2e_latency.p50),
        "e2e_p99_ms": _milliseconds(metrics.e2e_latency.p99),
        "prefill_step_p50_ms": _milliseconds(metrics.prefill_step_duration.p50),
        "prefill_step_p99_ms": _milliseconds(metrics.prefill_step_duration.p99),
        "decode_step_p50_ms": _milliseconds(metrics.decode_step_duration.p50),
        "decode_step_p99_ms": _milliseconds(metrics.decode_step_duration.p99),
        "preemptions": metrics.total_preemptions,
        "recomputed_tokens": metrics.total_recomputed_tokens,
        "prefix_hit_tokens": metrics.total_prefix_hit_tokens,
        "peak_used_kv_blocks": metrics.peak_used_kv_blocks,
    }


def environment_to_record(environment: dict) -> dict:
    """把 torch 等可能不是原生 str 的版本字段转成可 JSON 序列化值。"""
    return {
        key: value if value is None or isinstance(value, (str, int, float, bool)) else str(value)
        for key, value in environment.items()
    }


def dynamic_to_record(summary) -> dict:
    if summary is None:
        return {}
    return {
        "normal_decode_itl_p50_ms": summary.normal_decode_itl_p50_ms,
        "interrupted_decode_itl_p50_ms": summary.interrupted_decode_itl_p50_ms,
        "interrupted_decode_itl_max_ms": summary.interrupted_decode_itl_max_ms,
        "late_prefill_ttft_p50_ms": summary.late_prefill_ttft_p50_ms,
    }


def warmup_seed(base_seed: int, repeat: int, warmup_index: int) -> int:
    """为预热构造不与本次正式 seed 区间重叠的正整数 seed。"""
    return base_seed + repeat + WARMUP_SEED_OFFSET + warmup_index


def run_worker(args: argparse.Namespace) -> dict:
    """在一个干净 GPU 引擎中完成单个 workload 的同形状预热和正式重复。"""
    import torch

    from nanovllm import LLM, SamplingParams

    workload_name = args.worker_workload
    base_workload = build_workload_spec(workload_name, args.seed)
    llm = LLM(
        os.path.expanduser(args.model),
        enforce_eager=False,
        max_model_len=base_workload.max_model_len,
        max_num_batched_tokens=base_workload.max_num_batched_tokens,
        max_num_seqs=base_workload.max_num_seqs,
        num_kvcache_blocks=base_workload.num_kvcache_blocks,
        enable_stats=True,
        scheduler_policy=args.scheduler_policy,
        time_sliced_decode_steps=args.time_sliced_decode_steps,
    )
    atexit.unregister(llm.exit)
    runs = []
    try:
        # 使用独立的正整数 seed 域，既保持相同 shape/运行路径，又避免预热 Prompt
        # 与正式 seed=0..N 的 Prefix Cache 内容重合。预热结果不进入统计和报告。
        for warmup_index in range(args.warmup):
            current_warmup_seed = warmup_seed(
                args.seed, args.repeat, warmup_index
            )
            warmup_workload = build_workload_spec(
                workload_name, current_warmup_seed
            )
            torch.manual_seed(current_warmup_seed)
            run_workload_round(
                llm,
                warmup_workload,
                SamplingParams,
                synchronize=torch.cuda.synchronize,
                clock=time.perf_counter,
            )

        for run_index in range(args.repeat):
            run_seed = args.seed + run_index
            workload = build_workload_spec(workload_name, run_seed)
            torch.manual_seed(run_seed)
            result, sampling_params, elapsed = run_workload_round(
                llm,
                workload,
                SamplingParams,
                synchronize=torch.cuda.synchronize,
                clock=time.perf_counter,
            )
            metrics = llm.get_metrics_summary()
            assert metrics is not None
            output_lengths = [len(output["token_ids"]) for output in result.outputs]
            if output_lengths != workload.output_lengths:
                raise RuntimeError(
                    f"{workload_name} seed={run_seed} 输出长度 {output_lengths}，"
                    f"预期 {workload.output_lengths}"
                )
            output_tokens = sum(output_lengths)
            expected_tokens = sum(params.max_tokens for params in sampling_params)
            if output_tokens != expected_tokens:
                raise RuntimeError(
                    f"{workload_name} seed={run_seed} 输出 {output_tokens} token，"
                    f"预期 {expected_tokens}"
                )
            record = {
                "seed": run_seed,
                "elapsed_seconds": elapsed,
                "output_tokens": output_tokens,
                "output_lengths": output_lengths,
                "throughput": output_tokens / elapsed,
                **metrics_to_record(metrics),
                **dynamic_to_record(
                    summarize_dynamic_arrival(llm, workload, result)
                ),
            }
            runs.append(record)
    finally:
        llm.exit()

    return {
        "workload": workload_name,
        "workload_summary": format_workload(base_workload),
        "warmup": args.warmup,
        "repeat": args.repeat,
        "base_seed": args.seed,
        "scheduler_policy": args.scheduler_policy,
        "time_sliced_decode_steps": args.time_sliced_decode_steps,
        "runs": runs,
    }


def build_worker_command(script_path: Path, args, workload_name: str) -> list[str]:
    """构造只运行一个 workload 的独立子进程命令。"""
    scheduler_policy = getattr(args, "scheduler_policy", "prefill_first")
    time_sliced_decode_steps = getattr(args, "time_sliced_decode_steps", 4)
    return [
        sys.executable,
        str(script_path),
        "--model",
        args.model,
        "--warmup",
        str(args.warmup),
        "--repeat",
        str(args.repeat),
        "--seed",
        str(args.seed),
        "--scheduler-policy",
        scheduler_policy,
        "--time-sliced-decode-steps",
        str(time_sliced_decode_steps),
        "--worker-workload",
        workload_name,
    ]


def run_worker_process(script_path: Path, args, workload_name: str) -> dict:
    """运行干净子进程并读取结构化单场景结果。"""
    completed = subprocess.run(
        build_worker_command(script_path, args, workload_name),
        cwd=script_path.parent,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"workload {workload_name} 失败（exit={completed.returncode}）\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    raise RuntimeError(
        f"workload {workload_name} 没有输出结果标记 {RESULT_PREFIX}\n"
        f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )


SUMMARY_FIELDS = (
    "elapsed_seconds",
    "throughput",
    "ttft_p50_ms",
    "ttft_p99_ms",
    "itl_p50_ms",
    "itl_p99_ms",
    "itl_max_ms",
    "e2e_p50_ms",
    "e2e_p99_ms",
    "prefill_step_p50_ms",
    "decode_step_p50_ms",
    "preemptions",
    "recomputed_tokens",
    "prefix_hit_tokens",
    "peak_used_kv_blocks",
    "normal_decode_itl_p50_ms",
    "interrupted_decode_itl_p50_ms",
    "interrupted_decode_itl_max_ms",
    "late_prefill_ttft_p50_ms",
)

EXPECTED_PREFIX_HIT_TOKENS = {
    "short_prompt_long_decode": 0,
    "long_prompt_short_decode": 0,
    "mixed_lengths": 0,
    "shared_prefix_high_hit": 8192,
    "kv_pressure_preemption": 0,
    "decode_then_long_prefill": 0,
}


def summarize_result(result: dict) -> dict:
    """每个指标对多轮取中位数；不选择最好的一轮。"""
    runs = result["runs"]
    summary = {}
    for field in SUMMARY_FIELDS:
        values = [run[field] for run in runs if run.get(field) is not None]
        if values:
            summary[field] = median(values)
    return summary


def validate_result(result: dict, repeat: int, seed: int) -> None:
    """检查 worker 返回轮数、seed、完成状态和输出数量。"""
    runs = result["runs"]
    if len(runs) != repeat:
        raise ValueError(f"{result['workload']} 返回的正式轮数不正确")
    if [run["seed"] for run in runs] != list(range(seed, seed + repeat)):
        raise ValueError(f"{result['workload']} 的 seed 顺序不正确")
    for run in runs:
        if run["request_count"] != run["completed_request_count"]:
            raise ValueError(f"{result['workload']} 有未完成请求")
        if run["output_tokens"] <= 0:
            raise ValueError(f"{result['workload']} 没有正式输出 token")
        if sum(run["output_lengths"]) != run["output_tokens"]:
            raise ValueError(f"{result['workload']} 的输出长度与 token 总数不一致")
        expected_prefix_hits = EXPECTED_PREFIX_HIT_TOKENS[result["workload"]]
        if run["prefix_hit_tokens"] != expected_prefix_hits:
            raise ValueError(
                f"{result['workload']} 的 Prefix hit token="
                f"{run['prefix_hit_tokens']}，预期 {expected_prefix_hits}；"
                "可能存在预热或其他正式轮次留下的缓存污染"
            )


def _format_value(value, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def build_markdown_report(
    results: list[dict],
    *,
    model_path: str,
    warmup: int,
    repeat: int,
    seed: int,
    scheduler_policy: str = "prefill_first",
    time_sliced_decode_steps: int = 4,
    environment: dict,
) -> str:
    """把配置、每轮原始数据和多轮中位数生成可提交的中文基线报告。"""
    lines = [
        "# P0.4 可复现 Workload GPU 基线",
        "",
        "> 本文件由 `bench_suite.py` 生成。性能数字是实验基线，不作为 pytest 的硬断言。",
        "> 完整逐轮结构化指标保存在同名 `.json` 文件中。",
        "",
        "## 实验环境",
        "",
        f"- 日期：{environment['date']}",
        f"- Git commit：`{environment['commit']}`",
        f"- 工作区状态：{'dirty（包含未提交改动）' if environment.get('dirty') else 'clean'}",
        f"- GPU：{environment['gpu']}",
        f"- 模型：`{model_path}`",
        f"- Python：{environment['python']}",
        f"- PyTorch：{environment['torch']}",
        f"- CUDA：{environment['cuda']}",
        f"- 执行设置：CUDA Graph，stats 开启，scheduler_policy={scheduler_policy}，"
        f"time_sliced_decode_steps={time_sliced_decode_steps}，warmup={warmup}，repeat={repeat}，base_seed={seed}",
        "- 基线状态：valid（正式轮次已通过输出完整性与 Prefix Cache 隔离检查）",
        "",
        "## 六场景中位数",
        "",
        "| Workload | 吞吐 tok/s | TTFT P50/P99 ms | ITL P50/P99/Max ms | E2E P50/P99 ms | Prefill/Decode Step P50 ms | 抢占 | 重算 token | Prefix hit token | Peak KV Block |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        summary = summarize_result(result)
        lines.append(
            f"| `{result['workload']}` "
            f"| {_format_value(summary.get('throughput'))} "
            f"| {_format_value(summary.get('ttft_p50_ms'))}/{_format_value(summary.get('ttft_p99_ms'))} "
            f"| {_format_value(summary.get('itl_p50_ms'))}/{_format_value(summary.get('itl_p99_ms'))}/{_format_value(summary.get('itl_max_ms'))} "
            f"| {_format_value(summary.get('e2e_p50_ms'))}/{_format_value(summary.get('e2e_p99_ms'))} "
            f"| {_format_value(summary.get('prefill_step_p50_ms'))}/{_format_value(summary.get('decode_step_p50_ms'))} "
            f"| {_format_value(summary.get('preemptions'), 0)} "
            f"| {_format_value(summary.get('recomputed_tokens'), 0)} "
            f"| {_format_value(summary.get('prefix_hit_tokens'), 0)} "
            f"| {_format_value(summary.get('peak_used_kv_blocks'), 0)} |"
        )

    dynamic_result = next(
        (result for result in results if result["workload"] == "decode_then_long_prefill"),
        None,
    )
    if dynamic_result is not None:
        dynamic = summarize_result(dynamic_result)
        lines.extend(
            [
                "",
                "## 动态到达专项",
                "",
                f"- 正常 Decode ITL P50：{_format_value(dynamic.get('normal_decode_itl_p50_ms'))} ms",
                f"- 跨越迟到 Prefill 的 ITL P50/Max：{_format_value(dynamic.get('interrupted_decode_itl_p50_ms'))}/{_format_value(dynamic.get('interrupted_decode_itl_max_ms'))} ms",
                f"- 迟到长 Prompt TTFT P50：{_format_value(dynamic.get('late_prefill_ttft_p50_ms'))} ms",
            ]
        )

    lines.extend(["", "## 每轮原始数据", ""])
    for result in results:
        lines.extend(
            [
                f"### `{result['workload']}`",
                "",
                result["workload_summary"],
                "",
                "| Seed | 耗时 s | 吞吐 tok/s | TTFT P50/P99 ms | ITL P50/P99/Max ms | E2E P50/P99 ms |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for run in result["runs"]:
            lines.append(
                f"| {run['seed']} | {_format_value(run['elapsed_seconds'], 3)} "
                f"| {_format_value(run['throughput'])} "
                f"| {_format_value(run['ttft_p50_ms'])}/{_format_value(run['ttft_p99_ms'])} "
                f"| {_format_value(run['itl_p50_ms'])}/{_format_value(run['itl_p99_ms'])}/{_format_value(run['itl_max_ms'])} "
                f"| {_format_value(run['e2e_p50_ms'])}/{_format_value(run['e2e_p99_ms'])} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def collect_environment() -> dict:
    """在父进程收集可复现实验所需的软件、GPU 和 commit 信息。"""
    import torch

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.SubprocessError, OSError):
        commit = "unknown"
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()
        )
    except (subprocess.SubprocessError, OSError):
        dirty = None
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unavailable"
    return {
        "date": time.strftime("%Y-%m-%d"),
        "commit": commit,
        "dirty": dirty,
        "gpu": gpu,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "unavailable",
    }


def run_parent(args: argparse.Namespace) -> list[dict]:
    """顺序启动六个干净 worker，打印进度并写入统一 Markdown 报告。"""
    script_path = Path(__file__).resolve()
    workloads = args.workloads or list(OFFICIAL_WORKLOAD_NAMES)
    results = []
    print(
        f"P0.4 Benchmark Suite: policy={args.scheduler_policy}, workloads={len(workloads)}, warmup={args.warmup}, "
        f"repeat={args.repeat}, base_seed={args.seed}"
    )
    for index, workload_name in enumerate(workloads, start=1):
        print(f"[{index}/{len(workloads)}] Running {workload_name}...", flush=True)
        result = run_worker_process(script_path, args, workload_name)
        validate_result(result, args.repeat, args.seed)
        results.append(result)
        summary = summarize_result(result)
        print(
            f"[{index}/{len(workloads)}] {workload_name}: "
            f"throughput_median={summary['throughput']:.2f}tok/s, "
            f"TTFT_P99={summary['ttft_p99_ms']:.2f}ms, "
            f"ITL_P99={summary['itl_p99_ms']:.2f}ms"
        )

    report_path = Path(args.report).expanduser()
    if not report_path.is_absolute():
        report_path = script_path.parent / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    environment = environment_to_record(collect_environment())
    report_path.write_text(
        build_markdown_report(
            results,
            model_path=os.path.expanduser(args.model),
            warmup=args.warmup,
            repeat=args.repeat,
            seed=args.seed,
            scheduler_policy=args.scheduler_policy,
            time_sliced_decode_steps=args.time_sliced_decode_steps,
            environment=environment,
        ),
        encoding="utf-8",
    )
    json_path = report_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "valid": True,
                "environment": environment,
                "model": os.path.expanduser(args.model),
                "warmup": args.warmup,
                "repeat": args.repeat,
                "base_seed": args.seed,
                "scheduler_policy": args.scheduler_policy,
                "time_sliced_decode_steps": args.time_sliced_decode_steps,
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Baseline report written to {report_path}")
    print(f"Raw metrics written to {json_path}")
    return results


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.worker_workload is not None:
        print(RESULT_PREFIX + json.dumps(run_worker(args)), flush=True)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
