from types import SimpleNamespace

import pytest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_output import LegacySchedulerOutput
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.stats import EngineStats
from nanovllm.sampling_params import SamplingParams


def schedule_legacy(scheduler: Scheduler):
    output = scheduler.schedule()
    assert isinstance(output, LegacySchedulerOutput)
    assert output.total_num_scheduled_tokens == sum(
        output.num_scheduled_tokens.values()
    )
    return output.scheduled_seqs, output.is_prefill


class FakeClock:
    """测试专用的可控时钟：测试代码把时间拨到多少，统计代码就读到多少。"""

    def __init__(self, current_time: float = 0.0):
        self.current_time = current_time

    def set(self, current_time: float):
        self.current_time = current_time

    def __call__(self) -> float:
        return self.current_time


# 场景：请求在假时钟的 10 秒时通过 Scheduler.add() 加入 waiting 队列。
# 验证这里被定义为“到达 Scheduler 等待队列”的时刻；此时请求尚未被 schedule，
# 也没有生成任何真实 token，所以只有 queue_arrival_time 有值，其余事件仍为空。
def test_add_records_queue_arrival_time(monkeypatch: pytest.MonkeyPatch):
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    clock = FakeClock(current_time=10.0)
    stats = EngineStats(clock=clock)
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=99,
        kvcache_block_size=block_size,
        num_kvcache_blocks=4,
    )
    scheduler = Scheduler(config, stats=stats)
    seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=2),
    )

    scheduler.add(seq)

    metrics = stats.requests[seq.seq_id]
    assert metrics.queue_arrival_time == 10.0
    assert metrics.scheduled_times == []
    assert metrics.output_token_times == []
    assert metrics.request_finish_time is None
    assert metrics.first_scheduled_time is None
    assert metrics.first_token_time is None
    assert metrics.last_token_time is None
    assert metrics.queue_wait_time is None
    assert metrics.ttft is None
    assert metrics.itls == []
    assert metrics.e2e_latency is None


# 场景：6-token prompt 在 4-token budget 下分两轮完成 Prefill，然后再做一轮 Decode。
# 验证每次真正进入 scheduled_seqs 都会留下调度时间，因此三轮分别记录 1、3、6；
# 第一轮 Chunked Prefill 在时间 2 产生的临时候选 token 会被丢弃，不能误算成首 token；
# 真正输出发生在时间 5 和 8，由此得到 queue wait=1、TTFT=5、ITL=[3]、E2E=8。
def test_chunked_prefill_records_real_request_timeline(
    monkeypatch: pytest.MonkeyPatch,
):
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    clock = FakeClock(current_time=0.0)
    stats = EngineStats(clock=clock)
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=99,
        kvcache_block_size=block_size,
        num_kvcache_blocks=4,
    )
    scheduler = Scheduler(config, stats=stats)
    seq = Sequence(
        token_ids=[10, 11, 12, 13, 14, 15],
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler.add(seq)

    clock.set(1.0)
    first_seqs, is_prefill = schedule_legacy(scheduler)
    clock.set(2.0)
    scheduler.postprocess(first_seqs, token_ids=[89], is_prefill=is_prefill)

    # 第一段 Prefill 尚未覆盖完整 prompt，token 89 只是无效候选，不能进入指标。
    metrics = stats.requests[seq.seq_id]
    assert metrics.scheduled_times == [1.0]
    assert metrics.output_token_times == []
    assert metrics.first_token_time is None

    clock.set(3.0)
    second_seqs, is_prefill = schedule_legacy(scheduler)
    clock.set(5.0)
    scheduler.postprocess(second_seqs, token_ids=[90], is_prefill=is_prefill)

    clock.set(6.0)
    decode_seqs, is_prefill = schedule_legacy(scheduler)
    clock.set(8.0)
    scheduler.postprocess(decode_seqs, token_ids=[91], is_prefill=is_prefill)

    assert seq.status is SequenceStatus.FINISHED
    assert metrics.queue_arrival_time == 0.0
    assert metrics.scheduled_times == [1.0, 3.0, 6.0]
    assert metrics.output_token_times == [5.0, 8.0]
    assert metrics.request_finish_time == 8.0
    assert metrics.first_scheduled_time == 1.0
    assert metrics.first_token_time == 5.0
    assert metrics.last_token_time == 8.0
    assert metrics.queue_wait_time == 1.0
    assert metrics.ttft == 5.0
    assert metrics.itls == [3.0]
    assert metrics.e2e_latency == 8.0


# 场景：请求完成首轮 Prefill 后因为 KV Cache 紧张而被抢占，随后重新回到 waiting，
# 并在稍后再次被 Scheduler 选中。验证 preempt 本身不会清空或伪造时间事件：
# 原来的到达、调度和输出时间继续保留，重新被选中时只追加一条新的 scheduled 时间。
def test_preemption_preserves_request_timeline(monkeypatch: pytest.MonkeyPatch):
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    clock = FakeClock(current_time=10.0)
    stats = EngineStats(clock=clock)
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=8,
        eos=99,
        kvcache_block_size=block_size,
        num_kvcache_blocks=4,
    )
    scheduler = Scheduler(config, stats=stats)
    seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=3),
    )
    scheduler.add(seq)

    clock.set(12.0)
    prefill_seqs, is_prefill = schedule_legacy(scheduler)
    clock.set(15.0)
    scheduler.postprocess(prefill_seqs, token_ids=[90], is_prefill=is_prefill)

    clock.set(20.0)
    preempted_seq = scheduler.running.pop()
    scheduler.preempt(preempted_seq)

    metrics = stats.requests[seq.seq_id]
    assert metrics.scheduled_times == [12.0]
    assert metrics.output_token_times == [15.0]
    assert metrics.request_finish_time is None

    clock.set(25.0)
    recompute_seqs, is_prefill = schedule_legacy(scheduler)
    clock.set(30.0)
    scheduler.postprocess(recompute_seqs, token_ids=[91], is_prefill=is_prefill)

    assert seq.status is SequenceStatus.RUNNING
    assert metrics.queue_arrival_time == 10.0
    assert metrics.scheduled_times == [12.0, 25.0]
    assert metrics.output_token_times == [15.0, 30.0]
    assert metrics.request_finish_time is None
    assert metrics.queue_wait_time == 2.0
    assert metrics.ttft == 5.0
    assert metrics.itls == [15.0]


# 场景：请求的 max_tokens 还没有用完，但第二轮 Decode 采样到了 EOS=99。
# 验证 EOS token 本身仍是一个真实输出，因此会记录 output_token_time；同一个时间点
# 也被记为 request_finish_time，并据此正确计算最后一个 ITL 和端到端 E2E 延迟。
def test_eos_records_output_and_finish_time(monkeypatch: pytest.MonkeyPatch):
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    clock = FakeClock(current_time=100.0)
    stats = EngineStats(clock=clock)
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=99,
        kvcache_block_size=block_size,
        num_kvcache_blocks=4,
    )
    scheduler = Scheduler(config, stats=stats)
    seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=5),
    )
    scheduler.add(seq)

    clock.set(101.0)
    prefill_seqs, is_prefill = schedule_legacy(scheduler)
    clock.set(103.0)
    scheduler.postprocess(prefill_seqs, token_ids=[90], is_prefill=is_prefill)

    clock.set(104.0)
    decode_seqs, is_prefill = schedule_legacy(scheduler)
    clock.set(110.0)
    scheduler.postprocess(decode_seqs, token_ids=[99], is_prefill=is_prefill)

    metrics = stats.requests[seq.seq_id]
    assert seq.status is SequenceStatus.FINISHED
    assert seq.completion_token_ids == [90, 99]
    assert metrics.scheduled_times == [101.0, 104.0]
    assert metrics.output_token_times == [103.0, 110.0]
    assert metrics.request_finish_time == 110.0
    assert metrics.ttft == 3.0
    assert metrics.itls == [7.0]
    assert metrics.e2e_latency == 10.0


# 场景：像旧代码一样只传 config 创建 Scheduler，不提供 EngineStats。
# 验证关闭统计时完整 Prefill、token 追加、请求结束和 KV Block 释放仍照常发生；
# 这固定了“统计功能默认可关闭，并且不能改变原有调度与状态转换语义”的要求。
def test_scheduler_without_stats_keeps_original_behavior(
    monkeypatch: pytest.MonkeyPatch,
):
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=99,
        kvcache_block_size=block_size,
        num_kvcache_blocks=4,
    )
    scheduler = Scheduler(config)
    seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=1),
    )
    scheduler.add(seq)

    scheduled_seqs, is_prefill = schedule_legacy(scheduler)
    scheduler.postprocess(scheduled_seqs, token_ids=[90], is_prefill=is_prefill)

    assert scheduler.stats is None
    assert seq.completion_token_ids == [90]
    assert seq.status is SequenceStatus.FINISHED
    assert seq.block_table == []
    assert scheduler.is_finished() is True
