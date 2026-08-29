from types import SimpleNamespace

from nanovllm.engine.engine_output import (
    FinishedRequestOutput,
    SampledTokenOutput,
)
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.scheduler_output import UnifiedSchedulerOutput


class FakeModelRunner:
    def __init__(self, expected_input: UnifiedSchedulerOutput, token_ids: list[int]):
        self.expected_input = expected_input
        self.token_ids = token_ids

    def call(self, method: str, scheduler_output: UnifiedSchedulerOutput):
        assert method == "run"
        assert scheduler_output is self.expected_input
        return self.token_ids


class FakeScheduler:
    def __init__(self, scheduler_output: UnifiedSchedulerOutput):
        self.scheduler_output = scheduler_output

    def schedule(self) -> UnifiedSchedulerOutput:
        return self.scheduler_output

    def postprocess(
        self,
        scheduler_output: UnifiedSchedulerOutput,
        token_ids: list[int],
    ) -> None:
        assert scheduler_output is self.scheduler_output
        for seq, token_id, should_sample in zip(
            scheduler_output.scheduled_seqs,
            token_ids,
            scheduler_output.should_sample,
            strict=True,
        ):
            seq.num_scheduled_tokens = 0
            if not should_sample:
                continue
            seq.completion_token_ids.append(token_id)
            if seq.finish_after_step:
                seq.is_finished = True


def make_seq(
    seq_id: int,
    num_scheduled_tokens: int,
    completion_token_ids: list[int] | None = None,
    finish_after_step: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        seq_id=seq_id,
        num_scheduled_tokens=num_scheduled_tokens,
        completion_token_ids=list(completion_token_ids or []),
        finish_after_step=finish_after_step,
        is_finished=False,
    )


def make_engine(
    scheduler_output: UnifiedSchedulerOutput,
    token_ids: list[int],
) -> LLMEngine:
    engine = LLMEngine.__new__(LLMEngine)
    engine.stats = None
    engine.scheduler = FakeScheduler(scheduler_output)
    engine.model_runner = FakeModelRunner(scheduler_output, token_ids)
    return engine


# 场景：同一 Step 调度 3-token Partial Prefill 和 1-token Decode。ModelRunner
# 为每个请求返回一个按位置对齐的 token，但 should_sample=False 的 Prefill 占位值
# 不能成为采样事件；Decode token 必须保留，同时两类计算量应分别记录为 3 和 1。
def test_step_output_filters_partial_prefill_and_records_decode_sample():
    partial_prefill_seq = make_seq(seq_id=10, num_scheduled_tokens=3)
    decode_seq = make_seq(seq_id=20, num_scheduled_tokens=1)
    scheduler_output = UnifiedSchedulerOutput(
        scheduled_seqs=[partial_prefill_seq, decode_seq],
        total_num_scheduled_tokens=4,
        should_sample=[False, True],
        is_prefilling=[True, False],
    )
    engine = make_engine(scheduler_output, token_ids=[999, 202])

    step_output = engine.step()

    assert step_output.sampled_tokens == [
        SampledTokenOutput(seq_id=20, token_id=202)
    ]
    assert step_output.finished_requests == []
    assert step_output.num_step_prefill_tokens == 3
    assert step_output.num_step_decode_tokens == 1
    assert step_output.total_num_tokens == 4
    assert partial_prefill_seq.completion_token_ids == []
    assert decode_seq.completion_token_ids == [202]


# 场景：Final Prefill 到达采样边界，并在 postprocess 后完成请求。验证本轮 token
# 同时出现在 sampled_tokens 中，而 finished_requests 保存包含历史 token 的最终结果。
def test_step_output_records_final_prefill_sample_and_finished_request():
    seq = make_seq(
        seq_id=30,
        num_scheduled_tokens=4,
        completion_token_ids=[250],
        finish_after_step=True,
    )
    scheduler_output = UnifiedSchedulerOutput(
        scheduled_seqs=[seq],
        total_num_scheduled_tokens=4,
        should_sample=[True],
        is_prefilling=[True],
    )
    engine = make_engine(scheduler_output, token_ids=[303])

    step_output = engine.step()

    assert step_output.sampled_tokens == [
        SampledTokenOutput(seq_id=30, token_id=303)
    ]
    assert step_output.finished_requests == [
        FinishedRequestOutput(
            seq_id=30,
            final_completion_token_ids=[250, 303],
        )
    ]
    assert step_output.num_step_prefill_tokens == 4
    assert step_output.num_step_decode_tokens == 0
    assert step_output.total_num_tokens == 4
