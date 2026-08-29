import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.stats import (
    EngineStats,
    RequestMetrics,
    StepMetrics,
    summarize_percentiles,
)


# 场景：把 0～100 这 101 个数逆序交给百分位汇总器。
# 验证汇总器会先排序，并按照固定的线性插值规则得到 P50=50、P95=95、P99=99；
# 同时保留样本数、最小值和最大值，避免 benchmark 只显示分位数而看不到数据范围。
def test_percentile_summary_sorts_values_and_uses_linear_interpolation():
    summary = summarize_percentiles(range(100, -1, -1))

    assert summary.count == 101
    assert summary.minimum == 0
    assert summary.p50 == 50
    assert summary.p95 == 95
    assert summary.p99 == 99
    assert summary.maximum == 100


# 场景：分别汇总“完全没有样本”和“只有一个样本”两种边界输入。
# 验证空样本返回 count=0 且所有统计值为 None，而不是伪造延迟 0；单样本的
# P50/P95/P99、最小值和最大值都等于该样本本身，不发生除零或索引越界。
def test_percentile_summary_handles_empty_and_single_value():
    empty = summarize_percentiles([])
    single = summarize_percentiles([7.5])

    assert empty.count == 0
    assert empty.minimum is None
    assert empty.p50 is None
    assert empty.p95 is None
    assert empty.p99 is None
    assert empty.maximum is None

    assert single.count == 1
    assert single.minimum == 7.5
    assert single.p50 == 7.5
    assert single.p95 == 7.5
    assert single.p99 == 7.5
    assert single.maximum == 7.5


# 场景：两个已完成请求分别产生 TTFT=4/5、ITL=[2,3]/[5] 和 E2E=9/10；
# 第三个请求只有 arrival，尚未被调度或输出。验证汇总器从 RequestMetrics 中取对
# 各类样本：TTFT 每请求一个，ITL 展开为 [2,3,5]，缺失指标不会被错误补成 0。
def test_engine_summary_collects_request_samples_without_inventing_missing_values():
    stats = EngineStats()
    stats.requests = {
        0: RequestMetrics(
            queue_arrival_time=0.0,
            scheduled_times=[1.0],
            output_token_times=[4.0, 6.0, 9.0],
            request_finish_time=9.0,
        ),
        1: RequestMetrics(
            queue_arrival_time=10.0,
            scheduled_times=[12.0],
            output_token_times=[15.0, 20.0],
            request_finish_time=20.0,
        ),
        2: RequestMetrics(queue_arrival_time=30.0),
    }

    summary = stats.summarize()

    assert summary.request_count == 3
    assert summary.completed_request_count == 2
    assert summary.queue_wait.count == 2
    assert summary.queue_wait.p50 == pytest.approx(1.5)
    assert summary.ttft.count == 2
    assert summary.ttft.p50 == pytest.approx(4.5)
    assert summary.itl.count == 3
    assert summary.itl.minimum == pytest.approx(2.0)
    assert summary.itl.p50 == pytest.approx(3.0)
    assert summary.itl.maximum == pytest.approx(5.0)
    assert summary.e2e_latency.count == 2
    assert summary.e2e_latency.p50 == pytest.approx(9.5)


# 场景：两个请求带有不同的抢占、重算和 Prefix Hit 累计值，同时构造两轮 Prefill
# 三轮 Decode Step 和一轮 Mixed Step。验证请求事件使用求和汇总，Step 耗时按
# Prefill/Decode/Mixed 分开计算百分位，并从所有 Step 的 postprocess 后快照中
# 取出 peak_used_kv_blocks=4。
def test_engine_summary_sums_cache_events_and_separates_step_types():
    stats = EngineStats()
    stats.requests = {
        0: RequestMetrics(
            queue_arrival_time=0.0,
            preemption_count=1,
            preempted_cached_tokens=6,
            released_block_references=2,
            freed_physical_blocks=1,
            recompute_steps=1,
            recomputed_tokens=2,
            prefix_hit_blocks=1,
            prefix_hit_tokens=4,
        ),
        1: RequestMetrics(
            queue_arrival_time=0.0,
            preemption_count=2,
            preempted_cached_tokens=10,
            released_block_references=4,
            freed_physical_blocks=3,
            recompute_steps=2,
            recomputed_tokens=6,
            prefix_hit_blocks=2,
            prefix_hit_tokens=8,
        ),
    }
    stats.steps = [
        StepMetrics(0.0, 2.0, 1, 4, 0, used_kv_blocks=1),
        StepMetrics(2.0, 3.0, 1, 0, 1, used_kv_blocks=4),
        StepMetrics(3.0, 7.0, 1, 4, 0, used_kv_blocks=3),
        StepMetrics(7.0, 10.0, 2, 0, 2, used_kv_blocks=2),
        StepMetrics(10.0, 15.0, 1, 0, 1, used_kv_blocks=2),
        StepMetrics(15.0, 17.0, 2, 3, 2, used_kv_blocks=3),
    ]

    summary = stats.summarize()

    assert summary.step_count == 6
    assert summary.prefill_step_count == 2
    assert summary.decode_step_count == 3
    assert summary.mixed_step_count == 1
    assert summary.prefill_step_duration.count == 2
    assert summary.prefill_step_duration.p50 == pytest.approx(3.0)
    assert summary.decode_step_duration.count == 3
    assert summary.decode_step_duration.p50 == pytest.approx(3.0)
    assert summary.mixed_step_duration.count == 1
    assert summary.mixed_step_duration.p50 == pytest.approx(2.0)
    assert summary.peak_used_kv_blocks == 4
    assert summary.total_preemptions == 3
    assert summary.total_preempted_cached_tokens == 16
    assert summary.total_released_block_references == 6
    assert summary.total_freed_physical_blocks == 4
    assert summary.total_recompute_steps == 3
    assert summary.total_recomputed_tokens == 8
    assert summary.total_prefix_hit_blocks == 3
    assert summary.total_prefix_hit_tokens == 12


# 场景：一个请求内部以秒记录 Queue Wait=0.05、TTFT=0.1、ITL=0.02、E2E=0.12，
# 并有一轮耗时 0.003 秒的 Prefill。验证 benchmark 文本把这些延迟统一转换成毫秒，
# 显示样本数和 P50/P95/P99，同时保留请求、Step、抢占、重算和 KV 峰值摘要。
def test_metrics_report_formats_internal_seconds_as_milliseconds():
    stats = EngineStats()
    stats.requests = {
        0: RequestMetrics(
            queue_arrival_time=0.0,
            scheduled_times=[0.05],
            output_token_times=[0.1, 0.12],
            request_finish_time=0.12,
            preemption_count=1,
            recomputed_tokens=2,
            prefix_hit_tokens=4,
        )
    }
    stats.steps = [
        StepMetrics(
            start_time=0.0,
            end_time=0.003,
            num_seqs=1,
            num_prefill_tokens=4,
            num_decode_tokens=0,
            used_kv_blocks=3,
        )
    ]

    report = stats.summarize().format()

    assert "Requests: 1 (completed=1)" in report
    assert "TTFT (ms): count=1 P50=100.00 P95=100.00 P99=100.00" in report
    assert "ITL (ms): count=1 P50=20.00 P95=20.00 P99=20.00" in report
    assert "Prefill Step (ms): count=1 P50=3.00 P95=3.00 P99=3.00" in report
    assert "Steps: 1 (prefill=1, decode=0, mixed=0)" in report
    assert "Mixed Step (ms): count=0" in report
    assert "Preemptions: 1" in report
    assert "Recomputed tokens: 2" in report
    assert "Prefix hit tokens: 4" in report
    assert "Peak used KV blocks: 3" in report


# 场景：跳过 LLMEngine 的模型初始化，只分别注入启用和关闭状态的 EngineStats。
# 验证上层可以通过 get_metrics_summary() 取得汇总结果；关闭 enable_stats 时返回 None，
# 不会为了构造一份空报告而重新开启统计或改变原来的无统计执行路径。
def test_llm_engine_exposes_summary_only_when_stats_are_enabled():
    engine = LLMEngine.__new__(LLMEngine)
    engine.stats = EngineStats()
    engine.stats.requests[0] = RequestMetrics(queue_arrival_time=0.0)

    summary = engine.get_metrics_summary()

    assert summary is not None
    assert summary.request_count == 1

    engine.stats = None
    assert engine.get_metrics_summary() is None
