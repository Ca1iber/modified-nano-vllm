import pytest

from bench_prefix_cache import (
    build_worker_command,
    parse_args,
    summarize_comparison,
    validate_comparison,
)


def make_run(
    seed: int,
    *,
    elapsed: float,
    throughput: float,
    prefill_computed_tokens: int,
    prefix_hit_blocks: int,
    prefix_hit_tokens: int,
) -> dict:
    """构造一轮满足公共请求口径的结果，让测试只突出正在检查的 A/B 字段。"""
    return {
        "seed": seed,
        "target_digest": f"same-targets-{seed}",
        "request_count": 16,
        "completed_request_count": 16,
        "total_output_tokens": 512,
        "elapsed_seconds": elapsed,
        "throughput": throughput,
        "prefill_steps": 1,
        "prefill_computed_tokens": prefill_computed_tokens,
        "prefix_hit_blocks": prefix_hit_blocks,
        "prefix_hit_tokens": prefix_hit_tokens,
        "ttft_p50_ms": elapsed * 100,
        "ttft_p99_ms": elapsed * 110,
        "e2e_p50_ms": elapsed * 1000,
    }


def make_comparison_results():
    """构造三轮 Miss/Hit 数据，数值故意无序以验证汇总真的取中位数。"""
    miss_times = [3.0, 1.0, 2.0]
    hit_times = [1.0, 0.5, 0.75]
    miss_runs = [
        make_run(
            seed,
            elapsed=elapsed,
            throughput=512 / elapsed,
            prefill_computed_tokens=8704,
            prefix_hit_blocks=0,
            prefix_hit_tokens=0,
        )
        for seed, elapsed in enumerate(miss_times, start=7)
    ]
    hit_runs = [
        make_run(
            seed,
            elapsed=elapsed,
            throughput=512 / elapsed,
            prefill_computed_tokens=512,
            prefix_hit_blocks=32,
            prefix_hit_tokens=8192,
        )
        for seed, elapsed in enumerate(hit_times, start=7)
    ]
    return {"mode": "miss", "runs": miss_runs}, {"mode": "hit", "runs": hit_runs}


# 场景：用户直接运行 A/B benchmark 时，应得到足以降低单轮 GPU 波动的五次重复；
# 同时允许显式修改 repeat、seed 和模型目录。验证默认值与自定义值均能正确解析，
# 防止父进程和内部 worker 因参数口径不同而实际运行了不同请求。
def test_prefix_cache_benchmark_args_define_reproducible_ab_runs():
    defaults = parse_args([])
    custom = parse_args(["--repeat", "3", "--seed", "7", "--model", "/model"])

    assert defaults.repeat == 5
    assert defaults.seed == 0
    assert defaults.worker_mode is None
    assert custom.repeat == 3
    assert custom.seed == 7
    assert custom.model == "/model"


# 场景：Miss 与 Hit 必须分别启动独立 Python 进程，避免先运行的一组把 Prefix Cache
# 留给后一组。验证构造出的 worker 命令携带相同模型、repeat 和 seed，唯一差异只能
# 是 miss/hit 模式，从命令层面固定“同一试卷、两个干净引擎”的实验设计。
def test_prefix_cache_ab_workers_only_differ_by_mode(tmp_path):
    script_path = tmp_path / "bench_prefix_cache.py"
    miss = build_worker_command(script_path, "/model", "miss", repeat=5, seed=10)
    hit = build_worker_command(script_path, "/model", "hit", repeat=5, seed=10)

    assert miss[:5] == hit[:5]
    assert miss[6:] == hit[6:]
    assert miss[5] == "miss"
    assert hit[5] == "hit"


# 场景：A/B 性能数字只有在输入和工作量口径正确时才有意义。验证公平性检查接受
# 同 seed、同 Target 摘要、16/16 完成的结果，并严格要求 Miss 实算 8704 token、
# 命中 0，而 Hit 实算 512 token、命中 32 Blocks/8192 token；耗时本身不做硬断言。
def test_prefix_cache_ab_validation_checks_inputs_and_prefill_work():
    miss_result, hit_result = make_comparison_results()

    validate_comparison(miss_result, hit_result, repeat=3, seed=7)

    hit_result["runs"][1]["target_digest"] = "different-targets"
    with pytest.raises(ValueError, match="完全相同的 Target token"):
        validate_comparison(miss_result, hit_result, repeat=3, seed=7)


# 场景：GPU benchmark 每组会产生多轮有波动的原始数据，最终不能挑最快的一轮。
# 验证汇总采用耗时、吞吐和延迟的中位数，并按“Miss 中位耗时 ÷ Hit 中位耗时”
# 计算加速比；测试不要求真实 GPU 每一轮 Hit 都更快，避免把噪声变成正确性失败。
def test_prefix_cache_ab_summary_uses_medians_and_reports_speedup():
    miss_result, hit_result = make_comparison_results()

    summary = summarize_comparison(miss_result, hit_result)

    assert summary["miss_time"] == 2.0
    assert summary["hit_time"] == 0.75
    assert summary["time_speedup"] == pytest.approx(2.0 / 0.75)
    assert summary["miss_throughput"] == pytest.approx(256.0)
    assert summary["hit_throughput"] == pytest.approx(512 / 0.75)
