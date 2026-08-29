from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler_output import UnifiedSchedulerOutput


def make_runner(token_ids: list[int]) -> ModelRunner:
    """绕过 CUDA/NCCL 初始化，只保留 run() 分发需要的假依赖。"""
    runner = object.__new__(ModelRunner)
    runner.rank = 0
    runner.prepare_prefill = Mock(
        return_value=("prefill_input_ids", "prefill_positions")
    )
    runner.prepare_decode = Mock(
        return_value=("decode_input_ids", "decode_positions")
    )
    runner.prepare_sample = Mock(return_value="temperatures")
    runner.run_model = Mock(return_value="logits")
    runner.sampler = Mock(
        return_value=SimpleNamespace(tolist=lambda: token_ids)
    )
    return runner


def make_scheduler_output(is_prefilling: list[bool]) -> UnifiedSchedulerOutput:
    seqs = [object() for _ in is_prefilling]
    return UnifiedSchedulerOutput(
        scheduled_seqs=seqs,
        total_num_scheduled_tokens=len(seqs),
        should_sample=[True] * len(seqs),
        is_prefilling=is_prefilling,
    )


# 场景：Unified 输入中的请求全部处于 Prefill。验证 run() 只复用旧的
# prepare_prefill() 路径，并把恢复出的同质 phase=True 传给 run_model()。
def test_run_dispatches_homogeneous_prefill_to_legacy_prepare_path():
    runner = make_runner(token_ids=[101, 102])
    scheduler_output = make_scheduler_output([True, True])

    token_ids = runner.run(scheduler_output)

    assert token_ids == [101, 102]
    runner.prepare_prefill.assert_called_once_with(
        scheduler_output.scheduled_seqs
    )
    runner.prepare_decode.assert_not_called()
    runner.run_model.assert_called_once_with(
        "prefill_input_ids",
        "prefill_positions",
        True,
    )


# 场景：Unified 输入中的请求全部处于 Decode。验证 run() 只复用旧的
# prepare_decode() 路径，并把恢复出的同质 phase=False 传给 run_model()。
def test_run_dispatches_homogeneous_decode_to_legacy_prepare_path():
    runner = make_runner(token_ids=[201, 202])
    scheduler_output = make_scheduler_output([False, False])

    token_ids = runner.run(scheduler_output)

    assert token_ids == [201, 202]
    runner.prepare_prefill.assert_not_called()
    runner.prepare_decode.assert_called_once_with(
        scheduler_output.scheduled_seqs
    )
    runner.run_model.assert_called_once_with(
        "decode_input_ids",
        "decode_positions",
        False,
    )


# 场景：同一个 Unified Batch 同时包含 Decode 与 Prefill。在真正的混合 metadata
# 和 Attention 路径完成前，run() 必须 fail closed，不能误走任意一条旧 prepare。
def test_run_rejects_mixed_batch_before_legacy_prepare_path():
    runner = make_runner(token_ids=[301, 302])
    scheduler_output = make_scheduler_output([False, True])

    with pytest.raises(
        NotImplementedError,
        match="暂不支持混合 Prefill/Decode batch",
    ):
        runner.run(scheduler_output)

    runner.prepare_prefill.assert_not_called()
    runner.prepare_decode.assert_not_called()
    runner.run_model.assert_not_called()
