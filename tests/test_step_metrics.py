from types import SimpleNamespace

import pytest

from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler_output import (
    LegacySchedulerOutput,
    UnifiedSchedulerOutput,
)
from nanovllm.engine.stats import EngineStats
from nanovllm.sampling_params import SamplingParams


class FakeClock:
    """测试专用的可控时钟：测试代码把时间拨到多少，统计代码就读到多少。"""

    def __init__(self, current_time: float = 0.0):
        self.current_time = current_time

    def set(self, current_time: float):
        self.current_time = current_time

    def __call__(self) -> float:
        return self.current_time


class FakeScheduler:
    """返回预先准备好的 batch，并模拟 schedule 与 postprocess 发生的时间。"""

    def __init__(
        self,
        clock: FakeClock,
        seqs: list[SimpleNamespace],
        is_prefill: bool,
        schedule_time: float,
        postprocess_time: float,
        used_kv_blocks: int = 0,
        free_kv_blocks: int = 0,
    ):
        self.clock = clock
        self.seqs = seqs
        self.is_prefill = is_prefill
        self.schedule_time = schedule_time
        self.postprocess_time = postprocess_time
        self.last_scheduler_output = None
        self.postprocessed_output = None
        # LLMEngine.step() 会在 postprocess 后读取 BlockManager 的 used/free 数量。
        self.block_manager = SimpleNamespace(
            used_block_ids=set(range(used_kv_blocks)),
            free_block_ids=set(
                range(used_kv_blocks, used_kv_blocks + free_kv_blocks)
            ),
        )

    def schedule(self):
        self.clock.set(self.schedule_time)
        self.last_scheduler_output = LegacySchedulerOutput(
            scheduled_seqs=self.seqs,
            total_num_scheduled_tokens=sum(
                seq.num_scheduled_tokens for seq in self.seqs
            ),
            is_prefill=self.is_prefill,
        )
        return self.last_scheduler_output

    def postprocess(self, scheduler_output, token_ids):
        self.postprocessed_output = scheduler_output
        assert scheduler_output.scheduled_seqs == self.seqs
        assert len(token_ids) == len(scheduler_output.scheduled_seqs)
        assert scheduler_output.is_prefill is self.is_prefill
        # 真实 Scheduler.postprocess() 也会把本轮 scheduled token 清零。
        for seq in scheduler_output.scheduled_seqs:
            seq.num_scheduled_tokens = 0
        self.clock.set(self.postprocess_time)


class FakeModelRunner:
    """不执行模型，只模拟 ModelRunner.run 完成并返回一个候选 token。"""

    def __init__(self, clock: FakeClock, run_time: float):
        self.clock = clock
        self.run_time = run_time
        self.last_scheduler_output = None

    def call(self, method, scheduler_output):
        assert method == "run"
        assert isinstance(scheduler_output, UnifiedSchedulerOutput)
        self.last_scheduler_output = scheduler_output
        self.clock.set(self.run_time)
        return [90] * len(scheduler_output.scheduled_seqs)


def make_test_engine(
    stats: EngineStats | None,
    scheduler: FakeScheduler,
    model_runner: FakeModelRunner,
) -> LLMEngine:
    """跳过加载真实模型的 __init__，只组装 step() 测试需要的三个对象。"""
    engine = LLMEngine.__new__(LLMEngine)
    engine.stats = stats
    engine.scheduler = scheduler
    engine.model_runner = model_runner
    return engine


# 场景：动态 benchmark 在不同 Engine Step 分批调用 add_request()，必须记住每个
# RequestSpec 最终对应哪个 Sequence，才能提取旧 Decode 请求和迟到 Prefill 请求的
# 独立时间线。验证 add_request() 不仅把新 Sequence 交给 Scheduler，还返回同一个
# seq_id；已有调用方忽略返回值时，原本的入队行为保持不变。
def test_add_request_returns_the_scheduled_sequence_id():
    added_seqs = []
    engine = LLMEngine.__new__(LLMEngine)
    engine.tokenizer = SimpleNamespace(encode=lambda prompt: [10, 11])
    engine.scheduler = SimpleNamespace(add=added_seqs.append)

    seq_id = engine.add_request(
        "prompt",
        SamplingParams(ignore_eos=True, max_tokens=2),
    )

    assert len(added_seqs) == 1
    assert added_seqs[0].prompt_token_ids == [10, 11]
    assert seq_id == added_seqs[0].seq_id


# 场景：Scheduler 保留 Legacy 输出用于自身 postprocess，但 ModelRunner 只能看到经
# to_unified() 适配后的执行协议。验证 Engine 不按具体 Output 类型分支，并且适配对象
# 与原始对象共享同一批 Sequence，不复制或修改请求状态。
def test_engine_adapts_legacy_output_for_runner_and_keeps_raw_for_postprocess():
    clock = FakeClock(current_time=1.0)
    seq = SimpleNamespace(
        seq_id=0,
        num_tokens=4,
        num_cached_tokens=0,
        num_scheduled_tokens=4,
        is_finished=False,
        completion_token_ids=[],
    )
    scheduler = FakeScheduler(
        clock=clock,
        seqs=[seq],
        is_prefill=True,
        schedule_time=2.0,
        postprocess_time=4.0,
    )
    model_runner = FakeModelRunner(clock=clock, run_time=3.0)
    engine = make_test_engine(None, scheduler, model_runner)

    engine.step()

    raw_output = scheduler.last_scheduler_output
    runner_output = model_runner.last_scheduler_output
    assert isinstance(raw_output, LegacySchedulerOutput)
    assert isinstance(runner_output, UnifiedSchedulerOutput)
    assert runner_output is not raw_output
    assert runner_output.scheduled_seqs is raw_output.scheduled_seqs
    assert runner_output.is_prefilling == [True]
    assert runner_output.should_sample == [True]
    assert scheduler.postprocessed_output is raw_output


# 场景：一次 Prefill step 同时调度两个请求，本轮分别计算 4 和 2 个 prompt token。
# 假时钟在 step() 调用前是 10，schedule 后是 11，模型返回时是 15，postprocess
# 完成后是 16。验证 StepMetrics 覆盖完整的 schedule → run → postprocess 区间，
# 并且 Prefill token 必须在 postprocess 把 num_scheduled_tokens 清零之前统计为 6。
def test_prefill_step_records_batch_size_tokens_and_duration():
    clock = FakeClock(current_time=10.0)
    stats = EngineStats(clock=clock)
    seq1 = SimpleNamespace(
        seq_id=0,
        num_tokens=4,
        num_cached_tokens=0,
        num_scheduled_tokens=4,
        is_finished=False,
        completion_token_ids=[],
    )
    seq2 = SimpleNamespace(
        seq_id=1,
        num_tokens=2,
        num_cached_tokens=0,
        num_scheduled_tokens=2,
        is_finished=False,
        completion_token_ids=[],
    )
    scheduler = FakeScheduler(
        clock=clock,
        seqs=[seq1, seq2],
        is_prefill=True,
        schedule_time=11.0,
        postprocess_time=16.0,
    )
    model_runner = FakeModelRunner(clock=clock, run_time=15.0)
    engine = make_test_engine(stats, scheduler, model_runner)

    step_output = engine.step()

    assert step_output.finished_requests == []
    assert step_output.num_step_prefill_tokens == 6
    assert step_output.num_step_decode_tokens == 0
    assert len(stats.steps) == 1
    step = stats.steps[0]
    assert step.start_time == 10.0
    assert step.end_time == 16.0
    assert step.duration == 6.0
    assert step.is_prefill_only is True
    assert step.is_decode_only is False
    assert step.is_mixed is False
    assert step.num_seqs == 2
    assert step.num_prefill_tokens == 6
    assert step.num_decode_tokens == 0
    assert step.num_tokens == 6
    assert step.tokens_per_second == 1.0
    assert step.num_preemptions == 0
    assert step.num_recomputed_tokens == 0
    assert step.num_prefix_hit_tokens == 0
    assert step.used_kv_blocks == 0
    assert step.free_kv_blocks == 0
    assert engine.get_step_metrics() == [step]


# 场景：一个 prompt 总共有 6 个 token，但本轮 token budget 只允许 Chunked Prefill
# 前 4 个。验证 StepMetrics 记录的是“这一轮实际安排的 4 个 token”，而不是请求的
# prompt 总长度 6；剩余 2 个 token 应由下一次 step 另外形成一条记录。
def test_chunked_prefill_step_records_only_current_chunk():
    clock = FakeClock(current_time=20.0)
    stats = EngineStats(clock=clock)
    seq = SimpleNamespace(
        seq_id=0,
        num_tokens=6,
        num_cached_tokens=0,
        num_scheduled_tokens=4,
        is_finished=False,
        completion_token_ids=[],
    )
    scheduler = FakeScheduler(
        clock=clock,
        seqs=[seq],
        is_prefill=True,
        schedule_time=20.5,
        postprocess_time=22.0,
    )
    model_runner = FakeModelRunner(clock=clock, run_time=21.5)
    engine = make_test_engine(stats, scheduler, model_runner)

    engine.step()

    step = stats.steps[0]
    assert seq.num_tokens == 6
    assert seq.num_scheduled_tokens == 0
    assert step.num_prefill_tokens == 4
    assert step.num_decode_tokens == 0
    assert step.num_tokens == 4
    assert step.num_seqs == 1
    assert step.is_prefill_only is True


# 场景：一次 Decode step 调度两个 running 请求，每个请求本轮只计算一个 token。
# 验证 StepMetrics 和 EngineStepOutput 都把本轮工作量记录为 2 个 Decode token，
# 不再通过负数返回值编码整个 Step 的阶段。
def test_decode_step_records_decode_token_count():
    clock = FakeClock(current_time=30.0)
    stats = EngineStats(clock=clock)
    seq1 = SimpleNamespace(
        seq_id=0,
        num_tokens=6,
        num_cached_tokens=5,
        num_scheduled_tokens=1,
        is_finished=False,
        completion_token_ids=[],
    )
    seq2 = SimpleNamespace(
        seq_id=1,
        num_tokens=6,
        num_cached_tokens=5,
        num_scheduled_tokens=1,
        is_finished=False,
        completion_token_ids=[],
    )
    scheduler = FakeScheduler(
        clock=clock,
        seqs=[seq1, seq2],
        is_prefill=False,
        schedule_time=31.0,
        postprocess_time=35.0,
    )
    model_runner = FakeModelRunner(clock=clock, run_time=34.0)
    engine = make_test_engine(stats, scheduler, model_runner)

    step_output = engine.step()

    step = stats.steps[0]
    assert step_output.num_step_prefill_tokens == 0
    assert step_output.num_step_decode_tokens == 2
    assert step.is_prefill_only is False
    assert step.is_decode_only is True
    assert step.is_mixed is False
    assert step.num_seqs == 2
    assert step.num_prefill_tokens == 0
    assert step.num_decode_tokens == 2
    assert step.num_tokens == 2
    assert step.duration == 5.0
    assert step.tokens_per_second == 0.4


# 场景：enable_stats=False 时 LLMEngine.stats 为 None，但 step() 仍应完成原来的
# schedule、模型调用和 postprocess，并返回本轮 EngineStepOutput；读取 StepMetrics
# 时只得到空列表，证明关闭统计不会改变既有执行语义。
def test_step_without_stats_keeps_original_behavior():
    clock = FakeClock(current_time=40.0)
    seq = SimpleNamespace(
        seq_id=0,
        num_tokens=3,
        num_cached_tokens=0,
        num_scheduled_tokens=3,
        is_finished=False,
        completion_token_ids=[],
    )
    scheduler = FakeScheduler(
        clock=clock,
        seqs=[seq],
        is_prefill=True,
        schedule_time=41.0,
        postprocess_time=43.0,
    )
    model_runner = FakeModelRunner(clock=clock, run_time=42.0)
    engine = make_test_engine(None, scheduler, model_runner)

    step_output = engine.step()

    assert step_output.finished_requests == []
    assert step_output.num_step_prefill_tokens == 3
    assert step_output.num_step_decode_tokens == 0
    assert engine.get_step_metrics() == []


# 场景：同一个 EngineStats 已经保存了一个请求和一轮 StepMetrics，然后开始下一批
# generate。验证 reset() 会同时清空 request 与 step 两套记录，避免上一批生成的
# 时间线和执行轮次混入下一批 benchmark。
def test_reset_clears_request_and_step_metrics():
    clock = FakeClock(current_time=50.0)
    stats = EngineStats(clock=clock)
    stats.record_arrival(seq_id=0)
    step_start_time = clock()
    clock.set(54.0)
    stats.record_step(
        start_time=step_start_time,
        num_seqs=1,
        num_prefill_tokens=4,
        num_decode_tokens=0,
    )

    assert len(stats.requests) == 1
    assert len(stats.steps) == 1

    stats.reset()

    assert stats.requests == {}
    assert stats.steps == []


# 场景：动态 benchmark 不走 generate()，需要显式划分正式统计批次。验证引擎空闲
# 时 reset_metrics() 会清空旧请求和 Step；一旦 Scheduler 仍有 waiting/running 请求，
# 就拒绝重置，避免把执行中的请求时间线删掉后让后续统计访问不存在的 seq_id。
def test_engine_reset_metrics_requires_an_idle_scheduler():
    stats = EngineStats()
    stats.record_arrival(seq_id=0)
    engine = LLMEngine.__new__(LLMEngine)
    engine.stats = stats
    engine.scheduler = SimpleNamespace(is_finished=lambda: True)

    engine.reset_metrics()

    assert stats.requests == {}
    assert stats.steps == []

    engine.scheduler = SimpleNamespace(is_finished=lambda: False)
    with pytest.raises(RuntimeError, match="只能在引擎没有 waiting/running 请求时重置指标"):
        engine.reset_metrics()
