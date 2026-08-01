import pytest

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


# 场景：4-token prompt 连续追加两个生成 token 90、91。
# 验证 append_token 同步更新完整 token 列表、总长度和 last_token，同时保持原 Prompt
# 边界不变，使 prompt_token_ids、completion_token_ids 和生成 token 数量能够正确切分。
def test_append_token_updates_token_views():
    """Appending output tokens updates counts without changing the prompt view."""
    seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=3),
    )

    assert seq.token_ids == [10, 11, 12, 13]
    assert len(seq) == 4
    assert seq.last_token == 13
    assert seq.num_prompt_tokens == 4
    assert seq.prompt_token_ids == [10, 11, 12, 13]
    assert seq.completion_token_ids == []
    assert seq.num_completion_tokens == 0

    seq.append_token(90)
    seq.append_token(91)

    assert seq.token_ids == [10, 11, 12, 13, 90, 91]
    assert len(seq) == 6
    assert seq.num_tokens == 6
    assert seq.last_token == 91
    # Prompt 长度在创建 Sequence 时固定，追加生成 token 后原 Prompt 仍保持不变。
    assert seq.num_prompt_tokens == 4
    assert seq.prompt_token_ids == [10, 11, 12, 13]
    # 只有 Prompt 之后追加的 90、91 属于最终生成结果。
    assert seq.completion_token_ids == [90, 91]
    assert seq.num_completion_tokens == 2


# 场景：block_size=4，依次观察 token 数从 4→5→8→9 时的逻辑 Block 布局。
# 验证恰好填满 Block 时不多分配，跨过边界时新增一个 Block，并确保 block(i)
# 和 last_block_num_tokens 对每个边界都返回正确的 token 分片与尾块长度。
def test_block_properties_across_boundaries(monkeypatch: pytest.MonkeyPatch):
    """Logical block count and slices stay correct around block boundaries."""
    monkeypatch.setattr(Sequence, "block_size", 4)
    seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=8),
    )

    # 4 个 token 恰好填满一个 Block。
    assert seq.num_blocks == 1
    assert seq.last_block_num_tokens == 4
    assert seq.block(0) == [10, 11, 12, 13]

    # 第 5 个 token 跨入第二个 Block，尾块当前只有一个 token。
    seq.append_token(14)
    assert seq.num_blocks == 2
    assert seq.last_block_num_tokens == 1
    assert seq.block(0) == [10, 11, 12, 13]
    assert seq.block(1) == [14]

    # token 数达到 8 时，第二个 Block 也恰好填满，但仍然只需要两个 Block。
    seq.append_token(15)
    seq.append_token(16)
    seq.append_token(17)
    assert seq.num_blocks == 2
    assert seq.last_block_num_tokens == 4
    assert seq.block(1) == [14, 15, 16, 17]

    # 第 9 个 token 再次跨界，因此需要第三个 Block。
    seq.append_token(18)
    assert seq.num_blocks == 3
    assert seq.last_block_num_tokens == 1
    assert seq.block(2) == [18]


# 场景：同一个 Sequence 依次处于 WAITING、RUNNING 和 FINISHED。
# 验证 is_finished 只在 FINISHED 状态返回 True，避免引擎把等待中或运行中的请求
# 误判为已经完成，也避免完成请求继续留在生成循环中。
def test_is_finished_reflects_status():
    """Only the FINISHED status makes Sequence.is_finished true."""
    seq = Sequence(
        token_ids=[10],
        sampling_params=SamplingParams(max_tokens=2),
    )

    assert seq.status is SequenceStatus.WAITING
    assert seq.is_finished is False

    seq.status = SequenceStatus.RUNNING
    assert seq.is_finished is False

    seq.status = SequenceStatus.FINISHED
    assert seq.is_finished is True
