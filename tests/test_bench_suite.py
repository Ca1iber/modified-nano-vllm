from types import SimpleNamespace

import pytest

from bench_suite import (
    build_markdown_report,
    build_worker_command,
    environment_to_record,
    parse_args,
    summarize_result,
    validate_result,
    warmup_seed,
)
from bench_workloads import OFFICIAL_WORKLOAD_NAMES, build_workload_spec


def make_run(seed: int, scale: float = 1.0) -> dict:
    """构造一轮完整结构化结果，突出 suite 的汇总与报告逻辑。"""
    return {
        "seed": seed,
        "elapsed_seconds": 2.0 * scale,
        "output_tokens": 100,
        "output_lengths": [50, 50],
        "throughput": 50.0 / scale,
        "request_count": 2,
        "completed_request_count": 2,
        "step_count": 4,
        "prefill_steps": 1,
        "decode_steps": 3,
        "queue_wait_p50_ms": 1.0 * scale,
        "queue_wait_p99_ms": 2.0 * scale,
        "ttft_p50_ms": 10.0 * scale,
        "ttft_p99_ms": 12.0 * scale,
        "itl_p50_ms": 3.0 * scale,
        "itl_p99_ms": 4.0 * scale,
        "itl_max_ms": 5.0 * scale,
        "e2e_p50_ms": 20.0 * scale,
        "e2e_p99_ms": 22.0 * scale,
        "prefill_step_p50_ms": 8.0 * scale,
        "prefill_step_p99_ms": 9.0 * scale,
        "decode_step_p50_ms": 2.0 * scale,
        "decode_step_p99_ms": 2.5 * scale,
        "preemptions": 0,
        "recomputed_tokens": 0,
        "prefix_hit_tokens": 0,
        "peak_used_kv_blocks": 4,
    }


def make_result(workload: str = "short_prompt_long_decode") -> dict:
    return {
        "workload": workload,
        "workload_summary": f"Workload: name={workload}",
        "warmup": 1,
        "repeat": 3,
        "base_seed": 7,
        # 故意无序，让中位数不能碰巧等于第一轮或最后一轮。
        "runs": [make_run(7, 3.0), make_run(8, 1.0), make_run(9, 2.0)],
    }


# 场景：P0.4e 默认固定一次同形状预热、三次正式测量和六张正式试卷；用户也可以
# 只选部分 workload 做快速复测。验证 random_mixed 不进入正式列表，warmup 可设为
# 0，但 repeat 必须是正整数，避免生成没有正式数据的空报告。
def test_suite_args_define_six_official_workloads_and_fixed_repeats():
    defaults = parse_args([])
    selected = parse_args([
        "--warmup", "0",
        "--repeat", "5",
        "--seed", "10",
        "--workload", "mixed_lengths",
        "--workload", "decode_then_long_prefill",
    ])

    assert defaults.warmup == 1
    assert defaults.repeat == 3
    assert defaults.seed == 0
    assert defaults.workloads is None
    assert len(OFFICIAL_WORKLOAD_NAMES) == 6
    assert "random_mixed" not in OFFICIAL_WORKLOAD_NAMES
    assert selected.warmup == 0
    assert selected.repeat == 5
    assert selected.workloads == ["mixed_lengths", "decode_then_long_prefill"]

    with pytest.raises(SystemExit):
        parse_args(["--repeat", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--warmup", "-1"])
    with pytest.raises(SystemExit):
        parse_args(["--seed", "-1"])


# 场景：预热不能用 -1、-2 等负数 seed 与正式 seed 隔离，因为 Python 的
# Random(-1) 和 Random(1) 会生成相同 Prompt，使正式 seed=1 意外复用预热 KV。
# 验证预热使用独立的正整数 seed 域，且生成的长 Prompt 与三轮正式 Prompt 均不同。
def test_suite_warmup_seed_cannot_collide_with_formal_prefix_cache():
    formal_seeds = [0, 1, 2]
    selected_warmup_seed = warmup_seed(base_seed=0, repeat=3, warmup_index=0)
    warmup_prompt = build_workload_spec(
        "long_prompt_short_decode", selected_warmup_seed
    ).requests[0].prompt_token_ids

    assert selected_warmup_seed > max(formal_seeds)
    for formal_seed in formal_seeds:
        formal_prompt = build_workload_spec(
            "long_prompt_short_decode", formal_seed
        ).requests[0].prompt_token_ids
        assert warmup_prompt != formal_prompt


# 场景：每张试卷必须在独立 Python 子进程建立自己的模型、CUDA Graph、Prefix Cache
# 和 KV Block 容量。验证 worker 命令携带同一模型、warmup/repeat/seed，场景名只通过
# --worker-workload 传入，从命令边界杜绝前一场景的 GPU 状态污染后一场景。
def test_suite_worker_command_preserves_reproducible_settings(tmp_path):
    script = tmp_path / "bench_suite.py"
    args = SimpleNamespace(model="/model", warmup=1, repeat=3, seed=7)

    command = build_worker_command(script, args, "mixed_lengths")

    assert command[1] == str(script)
    assert command[command.index("--model") + 1] == "/model"
    assert command[command.index("--warmup") + 1] == "1"
    assert command[command.index("--repeat") + 1] == "3"
    assert command[command.index("--seed") + 1] == "7"
    assert command[-2:] == ["--worker-workload", "mixed_lengths"]


# 场景：三轮 GPU 数据会受时钟和频率波动，汇总不能挑最快一轮，也不能只保留最后
# 一轮。验证吞吐、TTFT、ITL、E2E 和缓存计数分别取三轮中位数；动态场景没有的
# 专项字段不会被伪造为 0，从而避免普通 workload 报告出现假的中断 ITL。
def test_suite_summary_uses_median_for_every_available_metric():
    summary = summarize_result(make_result())

    assert summary["elapsed_seconds"] == 4.0
    assert summary["throughput"] == 25.0
    assert summary["ttft_p50_ms"] == 20.0
    assert summary["itl_p99_ms"] == 8.0
    assert summary["e2e_p99_ms"] == 44.0
    assert summary["peak_used_kv_blocks"] == 4
    assert "interrupted_decode_itl_p50_ms" not in summary


# 场景：父进程只能接受完整的 worker 结果。验证正确的三轮 seed=7/8/9 且每轮所有
# 请求完成时通过；轮数、seed 顺序或完成数任一不符都明确报错，不能继续写出一份
# 表面完整但实验实际中断的基线报告。
def test_suite_validation_rejects_missing_or_incomplete_runs():
    result = make_result()
    validate_result(result, repeat=3, seed=7)

    result["runs"][1]["completed_request_count"] = 1
    with pytest.raises(ValueError, match="未完成请求"):
        validate_result(result, repeat=3, seed=7)


# 场景：除共享前缀试卷外，正式轮次不应命中上一轮或预热留下的 Prefix Cache。
# 验证一旦普通 workload 出现命中就拒绝报告，避免污染后的异常快数据被封存为基线。
def test_suite_validation_rejects_cross_round_prefix_cache_pollution():
    result = make_result()
    result["runs"][1]["prefix_hit_tokens"] = 256

    with pytest.raises(ValueError, match="缓存污染"):
        validate_result(result, repeat=3, seed=7)


# 场景：正式报告必须保存环境、统一中位数和每轮原始数据，不能只打印终端后丢失。
# 验证 Markdown 包含 commit/GPU、workload 汇总行和 seed=7/8/9 三轮记录；该测试只
# 检查稳定结构，不把真实 GPU 性能数字写成会随机器变化的硬断言。
def test_suite_report_contains_environment_medians_and_raw_runs():
    result = make_result()
    report = build_markdown_report(
        [result],
        model_path="/model",
        warmup=1,
        repeat=3,
        seed=7,
        environment={
            "date": "2026-08-14",
            "commit": "abcdef0",
            "dirty": True,
            "gpu": "Test GPU",
            "python": "3.10.20",
            "torch": "2.8.0",
            "cuda": "12.8",
        },
    )

    assert "# P0.4 可复现 Workload GPU 基线" in report
    assert "Git commit：`abcdef0`" in report
    assert "工作区状态：dirty（包含未提交改动）" in report
    assert "GPU：Test GPU" in report
    assert "基线状态：valid" in report
    assert "`short_prompt_long_decode` | 25.00" in report
    assert "| 7 | 6.000 | 16.67" in report
    assert "| 8 | 2.000 | 50.00" in report
    assert "| 9 | 4.000 | 25.00" in report


# 场景：部分 PyTorch 版本把 __version__ 暴露为 str 子类或专用对象。验证写 JSON 前
# 会把非原生标量统一转换成普通字符串，避免六个 workload 全部跑完后才在最终
# json.dumps() 阶段失败并丢失结构化结果。
def test_suite_environment_is_json_serializable():
    class VersionObject:
        def __str__(self):
            return "2.8.0+cu128"

    converted = environment_to_record({"torch": VersionObject(), "dirty": True})

    assert converted == {"torch": "2.8.0+cu128", "dirty": True}
