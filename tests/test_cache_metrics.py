from types import SimpleNamespace

import pytest

from nanovllm.engine.llm_engine import LLMEngine
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


class FakeModelRunner:
    """不执行模型，只为 LLMEngine.step() 返回固定的候选 token。"""

    def __init__(self, token_id: int = 90):
        self.token_id = token_id

    def call(self, method, seqs, is_prefill):
        assert method == "run"
        return [self.token_id] * len(seqs)


def make_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    num_blocks: int = 4,
    token_budget: int = 8,
    max_num_seqs: int = 2,
) -> tuple[Scheduler, EngineStats]:
    """创建 block_size=4 的纯 CPU Scheduler 和已开启的 EngineStats。"""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    stats = EngineStats()
    config = SimpleNamespace(
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=token_budget,
        eos=99,
        kvcache_block_size=block_size,
        num_kvcache_blocks=num_blocks,
    )
    return Scheduler(config, stats=stats), stats


def make_test_engine(scheduler: Scheduler, stats: EngineStats, token_id: int = 90):
    """跳过模型初始化，只组装一次 Engine step 需要的对象。"""
    engine = LLMEngine.__new__(LLMEngine)
    engine.scheduler = scheduler
    engine.stats = stats
    engine.model_runner = FakeModelRunner(token_id=token_id)
    return engine


# 场景：6-token Prompt 完成 Prefill 后拥有 6 个有效 KV token 和两个独占物理 Block，
# 随后整个请求被抢占。验证一次抢占会累计丢失的 KV token、解除的 Block 引用和
# 真正回到 free 队列的物理 Block；三者口径不同，但在没有共享 Block 时分别为 6、2、2。
def test_preemption_records_cached_tokens_references_and_freed_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler, stats = make_scheduler(monkeypatch)
    seq = Sequence(
        token_ids=[10, 11, 12, 13, 14, 15],
        sampling_params=SamplingParams(max_tokens=3),
    )
    scheduler.add(seq)

    prefill_seqs, is_prefill = schedule_legacy(scheduler)
    scheduler.postprocess(prefill_seqs, token_ids=[90], is_prefill=is_prefill)

    assert seq.num_cached_tokens == 6
    assert len(seq.block_table) == 2
    victim = scheduler.running.pop()
    scheduler.preempt(victim)

    metrics = stats.requests[seq.seq_id]
    assert metrics.preemption_count == 1
    assert metrics.preempted_cached_tokens == 6
    assert metrics.released_block_references == 2
    assert metrics.freed_physical_blocks == 2
    assert seq.status is SequenceStatus.WAITING
    assert seq.num_cached_tokens == 0
    assert seq.block_table == []
    assert len(scheduler.block_manager.free_block_ids) == 4


# 场景：两个请求共享第一个 Prefix Block，并各自拥有一个私有尾 Block；抢占第一个请求
# 会解除它的两个 Block 引用，但共享 Block 的 ref_count 只从 2 降到 1，不能回到 free。
# 验证 released_block_references=2，而真正释放的 freed_physical_blocks 只有私有尾块这 1 个。
def test_preemption_does_not_count_shared_block_as_freed(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler, stats = make_scheduler(monkeypatch)
    seq1 = Sequence(
        token_ids=[10, 11, 12, 13, 20, 21],
        sampling_params=SamplingParams(max_tokens=3),
    )
    seq2 = Sequence(
        token_ids=[10, 11, 12, 13, 30, 31],
        sampling_params=SamplingParams(max_tokens=3),
    )

    scheduler.add(seq1)
    prefill_seqs, is_prefill = schedule_legacy(scheduler)
    scheduler.postprocess(prefill_seqs, token_ids=[90], is_prefill=is_prefill)
    scheduler.add(seq2)
    prefill_seqs, is_prefill = schedule_legacy(scheduler)
    scheduler.postprocess(prefill_seqs, token_ids=[91], is_prefill=is_prefill)

    shared_block_id = seq1.block_table[0]
    seq1_private_block_id = seq1.block_table[1]
    assert seq2.block_table[0] == shared_block_id
    assert scheduler.block_manager.blocks[shared_block_id].ref_count == 2

    scheduler.running.remove(seq1)
    scheduler.preempt(seq1)

    metrics = stats.requests[seq1.seq_id]
    assert metrics.released_block_references == 2
    assert metrics.freed_physical_blocks == 1
    assert scheduler.block_manager.blocks[shared_block_id].ref_count == 1
    assert shared_block_id in scheduler.block_manager.used_block_ids
    assert shared_block_id not in scheduler.block_manager.free_block_ids
    assert scheduler.block_manager.blocks[seq1_private_block_id].ref_count == 0
    assert seq1_private_block_id in scheduler.block_manager.free_block_ids


# 场景：6-token Prompt 首次 Prefill 后被抢占，此时 0～5 共 6 个 token 的 KV 曾经算过。
# 恢复时命中第一个完整 Prefix Block（4 token），本轮实际调度 token 4、5、6 共 3 个；
# 其中只有 token 4、5 属于丢失后的重复计算，token 6 是输出 token 首次生成自己的 KV。
# 验证请求累计 prefix_hit=4、recomputed=2，StepMetrics 也记录相同事件和恢复后的 KV 快照。
def test_recompute_counts_only_previously_computed_uncached_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler, stats = make_scheduler(monkeypatch)
    seq = Sequence(
        token_ids=[10, 11, 12, 13, 14, 15],
        sampling_params=SamplingParams(max_tokens=3),
    )
    scheduler.add(seq)
    prefill_seqs, is_prefill = schedule_legacy(scheduler)
    scheduler.postprocess(prefill_seqs, token_ids=[90], is_prefill=is_prefill)

    scheduler.running.remove(seq)
    scheduler.preempt(seq)
    engine = make_test_engine(scheduler, stats, token_id=91)

    outputs, returned_num_tokens = engine.step()

    metrics = stats.requests[seq.seq_id]
    assert outputs == []
    assert returned_num_tokens == 3
    assert metrics.prefix_hit_blocks == 1
    assert metrics.prefix_hit_tokens == 4
    assert metrics.recompute_steps == 1
    assert metrics.recomputed_tokens == 2
    assert seq.num_cached_tokens == 7
    assert seq.completion_token_ids == [90, 91]

    step = stats.steps[-1]
    assert step.is_prefill is True
    assert step.num_tokens == 3
    # 抢占发生在本轮开始前；本轮只记录恢复产生的 Prefix Hit 和 Recompute。
    assert step.num_preemptions == 0
    assert step.num_prefix_hit_tokens == 4
    assert step.num_recomputed_tokens == 2
    assert step.used_kv_blocks == 2
    assert step.free_kv_blocks == 2


# 场景：一个从未被抢占的 6-token Prompt 因 4-token budget 被拆成两轮 Prefill。
# 验证第二轮虽然再次走 Prefill 路径，但它计算的是尚未处理过的新 token，不能因为
# “Prefill 不止一轮”就误记成 Recompute，也没有任何 Prefix Cache 命中或抢占事件。
def test_chunked_prefill_is_not_recompute(monkeypatch: pytest.MonkeyPatch):
    scheduler, stats = make_scheduler(monkeypatch, token_budget=4)
    seq = Sequence(
        token_ids=[10, 11, 12, 13, 14, 15],
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler.add(seq)

    first_seqs, is_prefill = schedule_legacy(scheduler)
    scheduler.postprocess(first_seqs, token_ids=[89], is_prefill=is_prefill)
    second_seqs, is_prefill = schedule_legacy(scheduler)
    scheduler.postprocess(second_seqs, token_ids=[90], is_prefill=is_prefill)

    metrics = stats.requests[seq.seq_id]
    assert metrics.preemption_count == 0
    assert metrics.recompute_steps == 0
    assert metrics.recomputed_tokens == 0
    assert metrics.prefix_hit_blocks == 0
    assert metrics.prefix_hit_tokens == 0


# 场景：两个请求各占一个 Block，物理池已经用满；下一轮 Decode 中第一个请求恰好
# 需要跨入新 Block，因此 Scheduler 自动抢占 running 队尾请求，并把释放出的 Block
# 立即分配给当前请求。验证该 Decode Step 记录一次抢占，同时结束快照仍是 used=2、free=0。
def test_decode_step_records_automatic_preemption_and_kv_snapshot(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler, stats = make_scheduler(
        monkeypatch,
        num_blocks=2,
        token_budget=8,
        max_num_seqs=2,
    )
    seq1 = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=3),
    )
    seq2 = Sequence(
        token_ids=[20, 21, 22, 23],
        sampling_params=SamplingParams(max_tokens=3),
    )
    scheduler.add(seq1)
    scheduler.add(seq2)
    prefill_seqs, is_prefill = schedule_legacy(scheduler)
    scheduler.postprocess(prefill_seqs, token_ids=[90, 91], is_prefill=is_prefill)
    assert len(scheduler.block_manager.free_block_ids) == 0

    engine = make_test_engine(scheduler, stats, token_id=92)
    _, returned_num_tokens = engine.step()

    seq2_metrics = stats.requests[seq2.seq_id]
    step = stats.steps[-1]
    assert returned_num_tokens == -1
    assert seq2.status is SequenceStatus.WAITING
    assert seq2_metrics.preemption_count == 1
    assert seq2_metrics.preempted_cached_tokens == 4
    assert seq2_metrics.freed_physical_blocks == 1
    assert step.is_prefill is False
    assert step.num_preemptions == 1
    assert step.num_recomputed_tokens == 0
    assert step.num_prefix_hit_tokens == 0
    assert step.used_kv_blocks == 2
    assert step.free_kv_blocks == 0


# 场景：max_tokens=1 的请求在首次完整 Prefill 采样后立刻结束，postprocess 会释放
# 它持有的全部 KV Block。验证 StepMetrics 的 used/free 快照采集在资源释放之后，
# 因而看到 used=0、free=4，而不是模型执行期间仍被占用的中间状态。
def test_step_kv_snapshot_is_taken_after_finished_request_releases_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    scheduler, stats = make_scheduler(monkeypatch, num_blocks=4, token_budget=4)
    seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=1),
    )
    scheduler.add(seq)
    engine = make_test_engine(scheduler, stats, token_id=90)

    outputs, returned_num_tokens = engine.step()

    step = stats.steps[-1]
    assert outputs == [(seq.seq_id, [90])]
    assert returned_num_tokens == 4
    assert seq.status is SequenceStatus.FINISHED
    assert step.used_kv_blocks == 0
    assert step.free_kv_blocks == 4
