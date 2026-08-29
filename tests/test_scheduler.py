from types import SimpleNamespace

import pytest

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.scheduler_output import (
    LegacySchedulerOutput,
    UnifiedSchedulerOutput,
)
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def schedule_legacy(scheduler: Scheduler):
    output = scheduler.schedule()
    assert isinstance(output, LegacySchedulerOutput)
    assert output.total_num_scheduled_tokens == sum(
        seq.num_scheduled_tokens for seq in output.scheduled_seqs
    )
    return output


def schedule_unified(scheduler: Scheduler):
    output = scheduler.schedule()
    assert isinstance(output, UnifiedSchedulerOutput)
    assert output.total_num_scheduled_tokens == sum(
        seq.num_scheduled_tokens for seq in output.scheduled_seqs
    )
    assert (
        len(output.scheduled_seqs)
        == len(output.is_prefilling)
        == len(output.should_sample)
    )
    return output


def make_scheduler_config(**kwargs):
    config = dict(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=99,
        kvcache_block_size=256,
        num_kvcache_blocks=4,
    )
    config.update(kwargs)
    return SimpleNamespace(**config)


def test_unified_output_to_unified_returns_self():
    seq = Sequence([10, 11], SamplingParams(max_tokens=1))
    seq.num_scheduled_tokens = 2
    output = UnifiedSchedulerOutput(
        scheduled_seqs=[seq],
        total_num_scheduled_tokens=2,
        should_sample=[True],
        is_prefilling=[True],
    )

    assert output.to_unified() is output


@pytest.mark.parametrize(
    (
        "is_prefill",
        "num_cached_tokens",
        "num_scheduled_tokens",
        "expected_should_sample",
    ),
    [
        pytest.param(True, 0, 4, False, id="partial-prefill"),
        pytest.param(True, 4, 2, True, id="final-prefill-chunk"),
        pytest.param(False, 5, 1, True, id="decode"),
    ],
)
def test_legacy_output_to_unified_maps_phase_and_sampling_boundary(
    is_prefill: bool,
    num_cached_tokens: int,
    num_scheduled_tokens: int,
    expected_should_sample: bool,
):
    token_ids = (
        [10, 11, 12, 13, 14]
        if not is_prefill
        else [10, 11, 12, 13, 14, 15]
    )
    seq = Sequence(
        token_ids,
        SamplingParams(max_tokens=2),
    )
    if not is_prefill:
        seq.append_token(15)
    seq.num_cached_tokens = num_cached_tokens
    seq.num_scheduled_tokens = num_scheduled_tokens
    original_token_ids = list(seq.token_ids)
    original_status = seq.status
    output = LegacySchedulerOutput(
        scheduled_seqs=[seq],
        total_num_scheduled_tokens=num_scheduled_tokens,
        is_prefill=is_prefill,
    )

    converted = output.to_unified()

    assert converted is not output
    assert converted.scheduled_seqs[0] is seq
    assert converted.total_num_scheduled_tokens == num_scheduled_tokens
    assert converted.is_prefilling == [is_prefill]
    assert converted.should_sample == [expected_should_sample]
    assert output.is_prefill is is_prefill
    assert seq.num_cached_tokens == num_cached_tokens
    assert seq.num_scheduled_tokens == num_scheduled_tokens
    assert seq.token_ids == original_token_ids
    assert seq.status is original_status


def test_legacy_output_to_unified_keeps_per_request_sampling_boundaries():
    partial_seq = Sequence(
        [10, 11, 12, 13, 14, 15],
        SamplingParams(max_tokens=1),
    )
    partial_seq.num_scheduled_tokens = 4
    final_seq = Sequence(
        [20, 21, 22, 23],
        SamplingParams(max_tokens=1),
    )
    final_seq.num_scheduled_tokens = 4
    output = LegacySchedulerOutput(
        scheduled_seqs=[partial_seq, final_seq],
        total_num_scheduled_tokens=8,
        is_prefill=True,
    )

    converted = output.to_unified()

    assert converted.scheduled_seqs == [partial_seq, final_seq]
    assert converted.is_prefilling == [True, True]
    assert converted.should_sample == [False, True]
    assert (
        len(converted.scheduled_seqs)
        == len(converted.is_prefilling)
        == len(converted.should_sample)
    )


def test_scheduler_defaults_to_legacy_mode():
    scheduler = Scheduler(make_scheduler_config())
    seq = Sequence([10, 11], SamplingParams(max_tokens=1))
    scheduler.add(seq)

    output = scheduler.schedule()

    assert scheduler.scheduler_mode == "legacy"
    assert isinstance(output, LegacySchedulerOutput)
    assert output.is_prefill is True


def test_scheduler_rejects_unknown_mode():
    scheduler = Scheduler(make_scheduler_config(scheduler_mode="unknown"))

    with pytest.raises(ValueError, match="不支持的 scheduler_mode"):
        scheduler.schedule()


def test_unified_mode_schedules_waiting_request():
    scheduler = Scheduler(make_scheduler_config(scheduler_mode="unified"))
    seq = Sequence([10, 11], SamplingParams(max_tokens=1))
    scheduler.add(seq)

    output = schedule_unified(scheduler)

    assert output.scheduled_seqs == [seq]
    assert output.total_num_scheduled_tokens == 2
    assert output.is_prefilling == [True]
    assert output.should_sample == [True]
    assert seq.status is SequenceStatus.RUNNING
    assert seq.is_prefill is True
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [seq]


def test_unified_mode_raises_when_no_request_can_be_scheduled():
    scheduler = Scheduler(make_scheduler_config(scheduler_mode="unified"))

    with pytest.raises(RuntimeError, match="没有可以调度的请求"):
        scheduler.schedule()


def test_unified_chunked_prefill_enters_running(
    monkeypatch: pytest.MonkeyPatch,
):
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    scheduler = Scheduler(
        make_scheduler_config(
            scheduler_mode="unified",
            max_num_batched_tokens=4,
            kvcache_block_size=block_size,
        )
    )
    seq = Sequence(
        token_ids=[10, 11, 12, 13, 14, 15],
        sampling_params=SamplingParams(max_tokens=1),
    )
    scheduler.add(seq)

    output = schedule_unified(scheduler)

    assert output.scheduled_seqs == [seq]
    assert output.total_num_scheduled_tokens == 4
    assert output.is_prefilling == [True]
    assert output.should_sample == [False]
    assert seq.num_cached_tokens == 0
    assert seq.num_scheduled_tokens == 4
    assert seq.status is SequenceStatus.RUNNING
    assert seq.is_prefill is True
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [seq]
    # Unified 首次分配只覆盖本轮 4-token chunk，不提前预留完整 Prompt。
    assert len(seq.block_table) == 1


def test_unified_prefix_cache_hit_schedules_only_uncached_suffix(
    monkeypatch: pytest.MonkeyPatch,
):
    """A unified prefill resumes after a cached block and hashes its suffix."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    scheduler = Scheduler(
        make_scheduler_config(
            scheduler_mode="unified",
            max_num_batched_tokens=4,
            kvcache_block_size=block_size,
            num_kvcache_blocks=4,
        )
    )

    cached_seq = Sequence(
        token_ids=[10, 11, 12, 13, 14],
        sampling_params=SamplingParams(max_tokens=1),
    )
    scheduler.block_manager.allocate(cached_seq)
    cached_seq.num_scheduled_tokens = block_size
    scheduler.block_manager.hash_blocks(cached_seq)
    shared_block_id = cached_seq.block_table[0]
    shared_block_hash = scheduler.block_manager.blocks[shared_block_id].hash
    cached_seq.num_cached_tokens += cached_seq.num_scheduled_tokens
    cached_seq.num_scheduled_tokens = 0
    scheduler.block_manager.deallocate(cached_seq)

    assert shared_block_hash != -1
    assert shared_block_id in scheduler.block_manager.free_block_ids
    assert scheduler.block_manager.hash_to_block_id[shared_block_hash] == shared_block_id

    seq = Sequence(
        token_ids=[10, 11, 12, 13, 20, 21, 22, 23],
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler.add(seq)

    output = schedule_unified(scheduler)

    assert output.scheduled_seqs == [seq]
    assert output.total_num_scheduled_tokens == block_size
    assert output.is_prefilling == [True]
    assert output.should_sample == [True]
    assert seq.num_cached_tokens == block_size
    assert seq.num_scheduled_tokens == block_size
    assert seq.block_table[0] == shared_block_id
    assert scheduler.block_manager.blocks[shared_block_id].ref_count == 1

    suffix_block_id = seq.block_table[1]
    assert scheduler.block_manager.blocks[suffix_block_id].hash == -1

    scheduler.postprocess(output, token_ids=[90])

    assert seq.num_cached_tokens == 2 * block_size
    assert seq.num_scheduled_tokens == 0
    assert seq.completion_token_ids == [90]
    assert seq.is_prefill is False
    suffix_block_hash = scheduler.block_manager.blocks[suffix_block_id].hash
    assert suffix_block_hash != -1
    assert scheduler.block_manager.hash_to_block_id[suffix_block_hash] == suffix_block_id


def test_unified_prefix_hit_partial_chunk_allocates_only_current_slots(
    monkeypatch: pytest.MonkeyPatch,
):
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    scheduler = Scheduler(
        make_scheduler_config(
            scheduler_mode="unified",
            max_num_batched_tokens=3,
            kvcache_block_size=block_size,
            num_kvcache_blocks=4,
        )
    )

    cached_seq = Sequence(
        token_ids=[10, 11, 12, 13, 14],
        sampling_params=SamplingParams(max_tokens=1),
    )
    scheduler.block_manager.allocate(cached_seq)
    cached_seq.num_scheduled_tokens = block_size
    scheduler.block_manager.hash_blocks(cached_seq)
    shared_block_id = cached_seq.block_table[0]
    cached_seq.num_cached_tokens += cached_seq.num_scheduled_tokens
    cached_seq.num_scheduled_tokens = 0
    scheduler.block_manager.deallocate(cached_seq)

    seq = Sequence(
        token_ids=[10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33],
        sampling_params=SamplingParams(max_tokens=1),
    )
    scheduler.add(seq)

    output = schedule_unified(scheduler)

    assert output.scheduled_seqs == [seq]
    assert output.total_num_scheduled_tokens == 3
    assert output.is_prefilling == [True]
    assert output.should_sample == [False]
    assert seq.num_cached_tokens == block_size
    assert seq.num_scheduled_tokens == 3
    assert len(seq.block_table) == 2
    assert seq.block_table[0] == shared_block_id
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [seq]


def test_unified_skips_unallocatable_waiter_and_schedules_later_request(
    monkeypatch: pytest.MonkeyPatch,
):
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    scheduler = Scheduler(
        make_scheduler_config(
            scheduler_mode="unified",
            max_num_batched_tokens=8,
            kvcache_block_size=block_size,
            num_kvcache_blocks=1,
        )
    )
    blocked_seq = Sequence(
        token_ids=[10, 11, 12, 13, 14, 15, 16, 17],
        sampling_params=SamplingParams(max_tokens=1),
    )
    schedulable_seq = Sequence(
        token_ids=[20, 21, 22, 23],
        sampling_params=SamplingParams(max_tokens=1),
    )
    scheduler.add(blocked_seq)
    scheduler.add(schedulable_seq)

    output = schedule_unified(scheduler)

    assert output.scheduled_seqs == [schedulable_seq]
    assert output.total_num_scheduled_tokens == block_size
    assert blocked_seq.status is SequenceStatus.WAITING
    assert blocked_seq.block_table == []
    assert blocked_seq.num_cached_tokens == 0
    assert blocked_seq.num_scheduled_tokens == 0
    assert list(scheduler.waiting) == [blocked_seq]
    assert schedulable_seq.status is SequenceStatus.RUNNING
    assert len(schedulable_seq.block_table) == 1
    assert list(scheduler.running) == [schedulable_seq]


def test_unified_chunk_allocates_next_block_only_after_crossing_boundary(
    monkeypatch: pytest.MonkeyPatch,
):
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    scheduler = Scheduler(
        make_scheduler_config(
            scheduler_mode="unified",
            max_num_batched_tokens=2,
            kvcache_block_size=block_size,
            num_kvcache_blocks=3,
        )
    )
    seq = Sequence(
        token_ids=[10, 11, 12, 13, 14, 15],
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler.add(seq)

    first_output = schedule_unified(scheduler)
    assert first_output.total_num_scheduled_tokens == 2
    assert first_output.should_sample == [False]
    assert len(seq.block_table) == 1
    scheduler.postprocess(first_output, token_ids=[90])
    assert seq.num_cached_tokens == 2
    assert len(seq.block_table) == 1

    second_output = schedule_unified(scheduler)
    assert second_output.total_num_scheduled_tokens == 2
    assert second_output.should_sample == [False]
    assert len(seq.block_table) == 1
    scheduler.postprocess(second_output, token_ids=[91])
    assert seq.num_cached_tokens == block_size
    assert len(seq.block_table) == 1

    third_output = schedule_unified(scheduler)
    assert third_output.total_num_scheduled_tokens == 2
    assert third_output.should_sample == [True]
    assert len(seq.block_table) == 2


def test_unified_admits_waiting_prefill_after_running_decode(
    monkeypatch: pytest.MonkeyPatch,
):
    """A running decode and a new waiter share one unified token budget."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    scheduler = Scheduler(
        make_scheduler_config(
            scheduler_mode="unified",
            max_num_seqs=2,
            max_num_batched_tokens=3,
            kvcache_block_size=block_size,
            num_kvcache_blocks=4,
        )
    )

    decode_seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=3),
    )
    scheduler.block_manager.allocate(decode_seq)
    decode_seq.num_cached_tokens = 4
    decode_seq.append_token(90)
    decode_seq.status = SequenceStatus.RUNNING
    decode_seq.is_prefill = False
    scheduler.running.append(decode_seq)

    prefill_seq = Sequence(
        token_ids=[20, 21, 22, 23],
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler.add(prefill_seq)

    assert list(scheduler.running) == [decode_seq]
    assert list(scheduler.waiting) == [prefill_seq]

    output = schedule_unified(scheduler)

    assert output.scheduled_seqs == [decode_seq, prefill_seq]
    assert output.total_num_scheduled_tokens == 3
    assert output.is_prefilling == [False, True]
    assert output.should_sample == [True, False]
    assert decode_seq.num_scheduled_tokens == 1
    assert prefill_seq.num_scheduled_tokens == 2
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [decode_seq, prefill_seq]
    assert prefill_seq.status is SequenceStatus.RUNNING
    assert prefill_seq.is_prefill is True

    scheduler.postprocess(output, token_ids=[91, 92])

    assert decode_seq.num_cached_tokens == 5
    assert decode_seq.num_scheduled_tokens == 0
    assert decode_seq.completion_token_ids == [90, 91]
    assert decode_seq.is_prefill is False
    assert prefill_seq.num_cached_tokens == 2
    assert prefill_seq.num_scheduled_tokens == 0
    assert prefill_seq.completion_token_ids == []
    assert prefill_seq.is_prefill is True
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [decode_seq, prefill_seq]


def test_unified_schedules_prefill_and_decode_from_running(
    monkeypatch: pytest.MonkeyPatch,
):
    """Mixed running requests commit progress and sample independently."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    scheduler = Scheduler(
        make_scheduler_config(
            scheduler_mode="unified",
            max_num_seqs=2,
            max_num_batched_tokens=3,
            kvcache_block_size=block_size,
            num_kvcache_blocks=4,
        )
    )

    decode_seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=3),
    )
    scheduler.block_manager.allocate(decode_seq)
    decode_seq.num_cached_tokens = 4
    decode_seq.append_token(90)
    decode_seq.status = SequenceStatus.RUNNING
    decode_seq.is_prefill = False

    prefill_seq = Sequence(
        token_ids=[20, 21, 22, 23, 24, 25, 26, 27],
        sampling_params=SamplingParams(max_tokens=1),
    )
    scheduler.block_manager.allocate(prefill_seq)
    prefill_seq.num_cached_tokens = 4
    prefill_seq.status = SequenceStatus.RUNNING
    prefill_seq.is_prefill = True

    scheduler.running.extend([decode_seq, prefill_seq])

    output = schedule_unified(scheduler)

    assert output.scheduled_seqs == [decode_seq, prefill_seq]
    assert output.total_num_scheduled_tokens == 3
    assert decode_seq.num_scheduled_tokens == 1
    assert prefill_seq.num_scheduled_tokens == 2
    assert output.is_prefilling == [False, True]
    assert output.should_sample == [True, False]
    assert decode_seq.num_cached_tokens == 4
    assert prefill_seq.num_cached_tokens == 4
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [decode_seq, prefill_seq]

    scheduler.postprocess(output, token_ids=[92, 93])

    assert decode_seq.num_cached_tokens == 5
    assert decode_seq.num_scheduled_tokens == 0
    assert decode_seq.completion_token_ids == [90, 92]
    assert decode_seq.is_prefill is False
    assert decode_seq.status is SequenceStatus.RUNNING

    assert prefill_seq.num_cached_tokens == 6
    assert prefill_seq.num_scheduled_tokens == 0
    assert prefill_seq.completion_token_ids == []
    assert prefill_seq.is_prefill is True
    assert prefill_seq.status is SequenceStatus.RUNNING
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [decode_seq, prefill_seq]


def test_unified_final_prefill_chunk_samples_and_switches_to_decode(
    monkeypatch: pytest.MonkeyPatch,
):
    """The final prefill chunk accepts one token and enters decode mode."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    scheduler = Scheduler(
        make_scheduler_config(
            scheduler_mode="unified",
            max_num_batched_tokens=4,
            kvcache_block_size=block_size,
        )
    )
    seq = Sequence(
        token_ids=[10, 11, 12, 13, 14, 15],
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler.add(seq)

    first_output = schedule_unified(scheduler)

    assert first_output.scheduled_seqs == [seq]
    assert first_output.is_prefilling == [True]
    assert first_output.should_sample == [False]
    scheduler.postprocess(first_output, token_ids=[90])

    assert seq.num_cached_tokens == 4
    assert seq.num_scheduled_tokens == 0
    assert seq.num_tokens == 6
    assert seq.completion_token_ids == []
    assert seq.status is SequenceStatus.RUNNING
    assert seq.is_prefill is True

    second_output = schedule_unified(scheduler)

    assert second_output.scheduled_seqs == [seq]
    assert second_output.total_num_scheduled_tokens == 2
    assert second_output.is_prefilling == [True]
    assert second_output.should_sample == [True]
    scheduler.postprocess(second_output, token_ids=[91])

    assert seq.num_cached_tokens == 6
    assert seq.num_scheduled_tokens == 0
    assert seq.num_tokens == 7
    assert seq.completion_token_ids == [91]
    assert seq.status is SequenceStatus.RUNNING
    assert seq.is_prefill is False
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [seq]


def test_unified_finished_request_releases_all_kv_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    """A final prefill sample can finish and fully release a request."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    scheduler = Scheduler(
        make_scheduler_config(
            scheduler_mode="unified",
            max_num_batched_tokens=4,
            kvcache_block_size=block_size,
            num_kvcache_blocks=2,
        )
    )
    seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=1),
    )
    scheduler.add(seq)

    output = schedule_unified(scheduler)

    assert output.scheduled_seqs == [seq]
    assert output.is_prefilling == [True]
    assert output.should_sample == [True]
    assert seq.num_cached_tokens == 0
    assert seq.num_scheduled_tokens == 4
    assert len(seq.block_table) == 1

    scheduler.postprocess(output, token_ids=[90])

    assert seq.completion_token_ids == [90]
    assert seq.status is SequenceStatus.FINISHED
    assert seq.is_prefill is False
    assert seq.num_cached_tokens == 0
    assert seq.num_scheduled_tokens == 0
    assert seq.block_table == []
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == []
    assert scheduler.block_manager.used_block_ids == set()
    assert len(scheduler.block_manager.free_block_ids) == 2
    assert all(block.ref_count == 0 for block in scheduler.block_manager.blocks)
    assert scheduler.is_finished() is True


@pytest.mark.parametrize(
    ("ignore_eos", "expected_status"),
    [
        (False, SequenceStatus.FINISHED),
        (True, SequenceStatus.RUNNING),
    ],
)
def test_unified_eos_respects_ignore_eos_and_kv_lifetime(
    monkeypatch: pytest.MonkeyPatch,
    ignore_eos: bool,
    expected_status: SequenceStatus,
):
    """Unified EOS handling either releases or retains the request KV blocks."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    scheduler = Scheduler(
        make_scheduler_config(
            scheduler_mode="unified",
            max_num_batched_tokens=block_size,
            kvcache_block_size=block_size,
            num_kvcache_blocks=4,
        )
    )
    seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(
            max_tokens=5,
            ignore_eos=ignore_eos,
        ),
    )
    scheduler.add(seq)

    prefill_output = schedule_unified(scheduler)
    assert prefill_output.should_sample == [True]
    scheduler.postprocess(prefill_output, token_ids=[90])

    decode_output = schedule_unified(scheduler)
    assert decode_output.is_prefilling == [False]
    assert decode_output.should_sample == [True]
    scheduler.postprocess(decode_output, token_ids=[99])

    assert seq.completion_token_ids == [90, 99]
    assert seq.num_completion_tokens < seq.max_tokens
    assert seq.num_scheduled_tokens == 0
    assert seq.status is expected_status

    if ignore_eos:
        allocated_block_ids = list(seq.block_table)
        assert seq.num_cached_tokens == 5
        assert list(scheduler.running) == [seq]
        assert scheduler.is_finished() is False
        assert scheduler.block_manager.used_block_ids == set(allocated_block_ids)
        assert all(
            scheduler.block_manager.blocks[block_id].ref_count == 1
            for block_id in allocated_block_ids
        )
    else:
        assert seq.num_cached_tokens == 0
        assert seq.block_table == []
        assert list(scheduler.running) == []
        assert scheduler.is_finished() is True
        assert scheduler.block_manager.used_block_ids == set()
        assert all(
            block.ref_count == 0
            for block in scheduler.block_manager.blocks
        )


def test_unified_matches_legacy_final_state(
    monkeypatch: pytest.MonkeyPatch,
):
    """Equivalent Legacy and Unified runs produce the same terminal state."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)

    def run_to_completion(scheduler_mode: str):
        scheduler = Scheduler(
            make_scheduler_config(
                scheduler_mode=scheduler_mode,
                max_num_batched_tokens=block_size,
                kvcache_block_size=block_size,
                num_kvcache_blocks=4,
            )
        )
        seq = Sequence(
            token_ids=[10, 11, 12, 13, 14, 15],
            sampling_params=SamplingParams(max_tokens=3),
        )
        scheduler.add(seq)

        num_steps = 0
        while not scheduler.is_finished():
            output = scheduler.schedule()
            token_ids = [
                90 + scheduled_seq.num_completion_tokens
                for scheduled_seq in output.scheduled_seqs
            ]
            scheduler.postprocess(output, token_ids)
            num_steps += 1
            assert num_steps < 10

        return {
            "completion_token_ids": seq.completion_token_ids,
            "status": seq.status,
            "num_cached_tokens": seq.num_cached_tokens,
            "num_scheduled_tokens": seq.num_scheduled_tokens,
            "block_table": list(seq.block_table),
            "waiting": list(scheduler.waiting),
            "running": list(scheduler.running),
            "used_block_ids": set(scheduler.block_manager.used_block_ids),
            "num_free_blocks": len(scheduler.block_manager.free_block_ids),
            "ref_counts": [
                block.ref_count for block in scheduler.block_manager.blocks
            ],
        }

    legacy_state = run_to_completion("legacy")
    unified_state = run_to_completion("unified")

    assert unified_state == legacy_state
    assert unified_state == {
        "completion_token_ids": [90, 91, 92],
        "status": SequenceStatus.FINISHED,
        "num_cached_tokens": 0,
        "num_scheduled_tokens": 0,
        "block_table": [],
        "waiting": [],
        "running": [],
        "used_block_ids": set(),
        "num_free_blocks": 4,
        "ref_counts": [0, 0, 0, 0],
    }


def test_unified_preemption_skips_waiting_and_prioritizes_victim(
    monkeypatch: pytest.MonkeyPatch,
):
    """A preempted running request stays ahead of an existing waiter."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    scheduler = Scheduler(
        make_scheduler_config(
            scheduler_mode="unified",
            max_num_seqs=3,
            max_num_batched_tokens=4,
            kvcache_block_size=block_size,
            num_kvcache_blocks=3,
        )
    )
    seq_a = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=3),
    )
    seq_b = Sequence(
        token_ids=[20, 21, 22, 23, 24, 25, 26, 27],
        sampling_params=SamplingParams(max_tokens=3),
    )
    seq_c = Sequence(
        token_ids=[30, 31, 32, 33],
        sampling_params=SamplingParams(max_tokens=3),
    )

    scheduler.block_manager.allocate(seq_a)
    seq_a.num_cached_tokens = 4
    seq_a.append_token(90)
    seq_a.status = SequenceStatus.RUNNING
    seq_a.is_prefill = False

    scheduler.block_manager.allocate(seq_b)
    seq_b.num_cached_tokens = 8
    seq_b.append_token(91)
    seq_b.status = SequenceStatus.RUNNING
    seq_b.is_prefill = False

    scheduler.running.extend([seq_a, seq_b])
    scheduler.waiting.append(seq_c)
    assert len(scheduler.block_manager.free_block_ids) == 0

    output = schedule_unified(scheduler)

    assert output.scheduled_seqs == [seq_a]
    assert output.total_num_scheduled_tokens == 1
    assert output.is_prefilling == [False]
    assert output.should_sample == [True]
    assert list(scheduler.running) == [seq_a]
    assert list(scheduler.waiting) == [seq_b, seq_c]
    assert seq_b.status is SequenceStatus.WAITING
    assert seq_b.is_prefill is True
    assert seq_b.num_cached_tokens == 0
    assert seq_b.block_table == []
    assert seq_c.status is SequenceStatus.WAITING
    assert seq_c.num_scheduled_tokens == 0
    assert seq_c.block_table == []
    assert len(scheduler.block_manager.free_block_ids) == 1


# 场景：6-token prompt 遇到 4-token budget，第一轮只能 Prefill 前 4 个 token。
# 验证请求仍留在 waiting，postprocess 只推进 cached_tokens、不记录临时采样 token；
# 同时检查 Scheduler 为完整 prompt 预留的两个 KV Block 及其引用计数。
def test_chunked_prefill_first_step_respects_token_budget(
    monkeypatch: pytest.MonkeyPatch,
):
    """A six-token prompt is split into a four-token first prefill chunk."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)

    # Scheduler only reads these fields, so the test does not need a model or GPU.
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=99,
        kvcache_block_size=block_size,
        num_kvcache_blocks=4,
    )
    scheduler = Scheduler(config)
    seq = Sequence(
        token_ids=[10, 11, 12, 13, 14, 15],
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler.add(seq)

    scheduled_seqs_output = schedule_legacy(scheduler)
    scheduled_seqs = scheduled_seqs_output.scheduled_seqs
    is_prefill = scheduled_seqs_output.is_prefill

    assert is_prefill is True
    assert scheduled_seqs == [seq]
    assert sum(item.num_scheduled_tokens for item in scheduled_seqs) == 4
    assert seq.num_scheduled_tokens == config.max_num_batched_tokens
    assert seq.num_cached_tokens == 0
    assert seq.status is SequenceStatus.WAITING
    assert list(scheduler.waiting) == [seq]
    assert list(scheduler.running) == []

    # The current BlockManager reserves every block needed by the prompt up front.
    assert len(seq.block_table) == 2
    assert scheduler.block_manager.used_block_ids == set(seq.block_table)
    assert all(
        scheduler.block_manager.blocks[block_id].ref_count == 1
        for block_id in seq.block_table
    )

    # An incomplete prefill chunk must advance cached progress but discard sampling.
    scheduler.postprocess(scheduled_seqs_output, token_ids=[90])

    assert seq.num_cached_tokens == 4
    assert seq.num_scheduled_tokens == 0
    assert seq.num_completion_tokens == 0
    assert seq.completion_token_ids == []
    assert seq.status is SequenceStatus.WAITING
    assert list(scheduler.waiting) == [seq]
    assert list(scheduler.running) == []


# 场景：同一个 6-token prompt 在 4-token budget 下需要连续两轮 Chunked Prefill。
# 验证两轮 postprocess 后 cached_tokens 按 0 → 4 → 6 单调推进，第一轮候选 token
# 被丢弃；第二轮完成整个 prompt 后，请求进入 running 并记录真正的首个生成 token。
def test_chunked_prefill_completes_across_multiple_steps(
    monkeypatch: pytest.MonkeyPatch,
):
    """A chunked prompt preserves progress and completes on its second step."""
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
        token_ids=[10, 11, 12, 13, 14, 15],
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler.add(seq)

    assert seq.num_cached_tokens == 0

    first_seqs_output = schedule_legacy(scheduler)
    first_seqs = first_seqs_output.scheduled_seqs
    is_prefill = first_seqs_output.is_prefill

    assert is_prefill is True
    assert first_seqs == [seq]
    assert seq.num_cached_tokens == 0
    assert seq.num_scheduled_tokens == 4
    assert seq.status is SequenceStatus.WAITING
    first_block_table = list(seq.block_table)

    # The first chunk is committed, but its sampled candidate is not generated output.
    scheduler.postprocess(first_seqs_output, token_ids=[90])

    assert seq.num_cached_tokens == 4
    assert seq.num_scheduled_tokens == 0
    assert seq.completion_token_ids == []
    assert seq.status is SequenceStatus.WAITING
    assert list(scheduler.waiting) == [seq]
    assert list(scheduler.running) == []

    second_seqs_output = schedule_legacy(scheduler)
    second_seqs = second_seqs_output.scheduled_seqs
    is_prefill = second_seqs_output.is_prefill

    assert is_prefill is True
    assert second_seqs == [seq]
    assert seq.num_cached_tokens == 4
    assert seq.num_scheduled_tokens == 2
    assert seq.status is SequenceStatus.RUNNING
    assert seq.block_table == first_block_table
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [seq]

    # The second chunk completes the prompt, so its sampled token is now accepted.
    scheduler.postprocess(second_seqs_output, token_ids=[91])

    assert seq.num_cached_tokens == 6
    assert seq.num_scheduled_tokens == 0
    assert seq.num_completion_tokens == 1
    assert seq.completion_token_ids == [91]
    assert seq.status is SequenceStatus.RUNNING
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [seq]


# 场景：4-token prompt 恰好装入 4-token budget，可以在一轮内完成整个 Prefill。
# 验证请求从 waiting 移到 running，并在 postprocess 中记录首个生成 token，
# 但尚未达到 max_tokens，因此仍保持 RUNNING。
def test_complete_prefill_moves_to_running_and_appends_first_token(
    monkeypatch: pytest.MonkeyPatch,
):
    """A prompt that fits the token budget finishes prefill in one step."""
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
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler.add(seq)

    scheduled_seqs_output = schedule_legacy(scheduler)
    scheduled_seqs = scheduled_seqs_output.scheduled_seqs
    is_prefill = scheduled_seqs_output.is_prefill

    assert is_prefill is True
    assert scheduled_seqs == [seq]
    assert seq.num_scheduled_tokens == 4
    assert seq.num_cached_tokens == 0
    assert seq.status is SequenceStatus.RUNNING
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [seq]

    # Simulate the model returning the first generated token after full prefill.
    scheduler.postprocess(scheduled_seqs_output, token_ids=[90])

    assert seq.num_cached_tokens == 4
    assert seq.num_scheduled_tokens == 0
    assert seq.num_completion_tokens == 1
    assert seq.completion_token_ids == [90]
    assert seq.status is SequenceStatus.RUNNING
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [seq]


# 场景：完整 Prefill 已生成 token 90，下一轮 Decode 再生成 token 91，达到 max_tokens=2。
# 验证请求变成 FINISHED、从 running 队列移除，并归还全部 KV Block，
# 最终 block_table 和 used_block_ids 均为空、所有 block 的 ref_count 均归零。
def test_decode_reaches_max_tokens_and_releases_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    """A finished decode leaves no sequence or allocated KV block behind."""
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
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler.add(seq)

    # Complete prefill and record the first generated token.
    prefill_seqs_output = schedule_legacy(scheduler)
    prefill_seqs = prefill_seqs_output.scheduled_seqs
    is_prefill = prefill_seqs_output.is_prefill
    scheduler.postprocess(prefill_seqs_output, token_ids=[90])

    assert seq.completion_token_ids == [90]
    assert seq.status is SequenceStatus.RUNNING

    # The next step is decode: token 90 is computed and token 91 is sampled.
    decode_seqs_output = schedule_legacy(scheduler)
    decode_seqs = decode_seqs_output.scheduled_seqs
    is_prefill = decode_seqs_output.is_prefill

    assert is_prefill is False
    assert decode_seqs == [seq]
    assert seq.num_scheduled_tokens == 1
    assert seq.num_cached_tokens == 4
    assert seq.is_prefill is False
    assert len(seq.block_table) == 2

    scheduler.postprocess(decode_seqs_output, token_ids=[91])

    assert seq.completion_token_ids == [90, 91]
    assert seq.num_completion_tokens == 2
    assert seq.status is SequenceStatus.FINISHED
    assert seq.num_scheduled_tokens == 0
    assert seq.num_cached_tokens == 0
    assert seq.block_table == []
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == []
    assert scheduler.block_manager.used_block_ids == set()
    assert len(scheduler.block_manager.free_block_ids) == config.num_kvcache_blocks
    assert all(block.ref_count == 0 for block in scheduler.block_manager.blocks)
    assert scheduler.is_finished() is True


# 场景：三个请求各有 2 个 prompt token，允许最多调度 3 个请求，但本轮总 budget 只有 4。
# 验证 Scheduler 只选择前两个请求、总 scheduled_tokens 恰好为 4，第三个请求继续 waiting；
# 因此停止调度的原因是 token budget 已用完，而不是触发了 max_num_seqs 限制。
def test_prefill_batch_does_not_exceed_token_budget_with_multiple_sequences(
    monkeypatch: pytest.MonkeyPatch,
):
    """A multi-request prefill batch shares one global token budget."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)

    config = SimpleNamespace(
        max_num_seqs=3,
        max_num_batched_tokens=4,
        eos=99,
        kvcache_block_size=block_size,
        num_kvcache_blocks=6,
    )
    scheduler = Scheduler(config)
    seq1 = Sequence(
        token_ids=[10, 11],
        sampling_params=SamplingParams(max_tokens=2),
    )
    seq2 = Sequence(
        token_ids=[20, 21],
        sampling_params=SamplingParams(max_tokens=2),
    )
    seq3 = Sequence(
        token_ids=[30, 31],
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler.add(seq1)
    scheduler.add(seq2)
    scheduler.add(seq3)

    scheduled_seqs_output = schedule_legacy(scheduler)
    scheduled_seqs = scheduled_seqs_output.scheduled_seqs
    is_prefill = scheduled_seqs_output.is_prefill

    total_scheduled_tokens = sum(
        seq.num_scheduled_tokens for seq in scheduled_seqs
    )
    assert is_prefill is True
    assert scheduled_seqs == [seq1, seq2]
    assert total_scheduled_tokens == 4
    assert total_scheduled_tokens <= config.max_num_batched_tokens
    assert seq1.num_scheduled_tokens == 2
    assert seq2.num_scheduled_tokens == 2
    assert seq3.num_scheduled_tokens == 0
    assert seq1.status is SequenceStatus.RUNNING
    assert seq2.status is SequenceStatus.RUNNING
    assert seq3.status is SequenceStatus.WAITING
    assert list(scheduler.running) == [seq1, seq2]
    assert list(scheduler.waiting) == [seq3]
    assert seq3.block_table == []


# 场景：请求 A 已完成 Prefill、正在 running 等待 Decode，此时长 Prompt 请求 B 进入 waiting。
# 验证当前 prefill_first 调度会优先选择 B 的 4-token Prefill chunk，本轮不调度 A；
# 该行为将作为未来实现 decode_first 和 time_sliced 策略时的默认基线。
def test_prefill_first_prioritizes_waiting_prompt_over_running_decode(
    monkeypatch: pytest.MonkeyPatch,
):
    """A schedulable waiting prompt takes priority over a running decode."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)

    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=99,
        kvcache_block_size=block_size,
        num_kvcache_blocks=6,
    )
    scheduler = Scheduler(config)
    decode_seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=3),
    )
    scheduler.add(decode_seq)

    # Move request A into running and give it its first generated token.
    prefill_seqs_output = schedule_legacy(scheduler)
    prefill_seqs = prefill_seqs_output.scheduled_seqs
    is_prefill = prefill_seqs_output.is_prefill
    scheduler.postprocess(prefill_seqs_output, token_ids=[90])

    assert decode_seq.status is SequenceStatus.RUNNING
    assert decode_seq.completion_token_ids == [90]
    assert decode_seq.num_scheduled_tokens == 0

    prefill_seq = Sequence(
        token_ids=[20, 21, 22, 23, 24, 25],
        sampling_params=SamplingParams(max_tokens=2),
    )
    scheduler.add(prefill_seq)

    assert list(scheduler.running) == [decode_seq]
    assert list(scheduler.waiting) == [prefill_seq]

    scheduled_seqs_output = schedule_legacy(scheduler)
    scheduled_seqs = scheduled_seqs_output.scheduled_seqs
    is_prefill = scheduled_seqs_output.is_prefill

    assert is_prefill is True
    assert scheduled_seqs == [prefill_seq]
    assert prefill_seq.num_scheduled_tokens == 4
    assert prefill_seq.status is SequenceStatus.WAITING
    assert decode_seq.num_scheduled_tokens == 0
    assert decode_seq.status is SequenceStatus.RUNNING
    assert list(scheduler.waiting) == [prefill_seq]
    assert list(scheduler.running) == [decode_seq]


# 场景：已有请求正在 Decode 时，又加入一个新的 waiting 请求。
# 验证 decode_first 会先继续调度 running 请求；只有 running 为空时，才回退到 Prefill。
def test_decode_first_prioritizes_running_decode_over_waiting_prefill(
    monkeypatch: pytest.MonkeyPatch,
):
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=99,
        kvcache_block_size=block_size,
        num_kvcache_blocks=6,
        scheduler_policy="decode_first",
        time_sliced_decode_steps=4,
    )
    scheduler = Scheduler(config)
    running_seq = Sequence([10, 11, 12, 13], SamplingParams(max_tokens=2))
    waiting_seq = Sequence([20, 21, 22, 23], SamplingParams(max_tokens=2))
    scheduler.add(running_seq)

    prefill_seqs_output = schedule_legacy(scheduler)
    prefill_seqs = prefill_seqs_output.scheduled_seqs
    is_prefill = prefill_seqs_output.is_prefill
    assert is_prefill is True  # 冷启动没有 running，只能先完成 Prefill。
    scheduler.postprocess(prefill_seqs_output, [90])
    scheduler.add(waiting_seq)

    decode_seqs_output = schedule_legacy(scheduler)
    decode_seqs = decode_seqs_output.scheduled_seqs
    is_prefill = decode_seqs_output.is_prefill
    assert is_prefill is False
    assert decode_seqs == [running_seq]
    assert list(scheduler.waiting) == [waiting_seq]

    # running 请求完成后，decode_first 仍能回退到 waiting 队列完成 Prefill。
    scheduler.postprocess(decode_seqs_output, [91])
    prefill_seqs_output = schedule_legacy(scheduler)
    prefill_seqs = prefill_seqs_output.scheduled_seqs
    is_prefill = prefill_seqs_output.is_prefill
    assert is_prefill is True
    assert prefill_seqs == [waiting_seq]


# 场景：time_sliced 配置为连续 2 个 Decode Step 后让等待请求获得一次机会。
# 验证前两轮优先 Decode，达到配额后第三轮切换 Prefill，并在切换后清零计数器。
def test_time_sliced_switches_to_waiting_prefill_after_decode_quota(
    monkeypatch: pytest.MonkeyPatch,
):
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=99,
        kvcache_block_size=block_size,
        num_kvcache_blocks=6,
        scheduler_policy="time_sliced",
        time_sliced_decode_steps=2,
    )
    scheduler = Scheduler(config)
    running_seq = Sequence([10, 11, 12, 13], SamplingParams(max_tokens=8))
    waiting_seq = Sequence([20, 21, 22, 23], SamplingParams(max_tokens=2))
    scheduler.add(running_seq)
    prefill_seqs_output = schedule_legacy(scheduler)
    prefill_seqs = prefill_seqs_output.scheduled_seqs
    is_prefill = prefill_seqs_output.is_prefill
    scheduler.postprocess(prefill_seqs_output, [90])
    scheduler.add(waiting_seq)

    first_output = schedule_legacy(scheduler)
    first = first_output.scheduled_seqs
    is_prefill = first_output.is_prefill
    assert first == [running_seq]
    assert is_prefill is False
    assert scheduler.decode_steps_since_prefill == 1
    scheduler.postprocess(first_output, [91])

    second_output = schedule_legacy(scheduler)
    second = second_output.scheduled_seqs
    is_prefill = second_output.is_prefill
    assert second == [running_seq]
    assert is_prefill is False
    assert scheduler.decode_steps_since_prefill == 2
    scheduler.postprocess(second_output, [92])

    third_output = schedule_legacy(scheduler)
    third = third_output.scheduled_seqs
    is_prefill = third_output.is_prefill
    assert third == [waiting_seq]
    assert is_prefill is True
    assert scheduler.decode_steps_since_prefill == 0


# 场景：请求 A 已经进入 Decode 后，请求 B 才到达；分别使用三种调度策略把两条请求
# 一直运行到结束。策略只应改变 A、B 获得 Step 的先后顺序，不能丢失或重复输出 token。
# 验证两条请求最终都生成预期 token、进入 FINISHED，并从 waiting/running 中移除；同时
# 检查 block_table、used_block_ids 和全部 ref_count，确保不同策略都完整释放 KV 资源。
@pytest.mark.parametrize(
    "scheduler_policy",
    ["prefill_first", "decode_first", "time_sliced"],
)
def test_scheduler_policies_finish_same_outputs_and_release_all_kv_blocks(
    monkeypatch: pytest.MonkeyPatch,
    scheduler_policy: str,
):
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)
    config = SimpleNamespace(
        max_num_seqs=2,
        max_num_batched_tokens=4,
        eos=99,
        kvcache_block_size=block_size,
        num_kvcache_blocks=8,
        scheduler_policy=scheduler_policy,
        time_sliced_decode_steps=2,
    )
    scheduler = Scheduler(config)
    seq_a = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=4),
    )
    seq_b = Sequence(
        token_ids=[20, 21, 22, 23],
        sampling_params=SamplingParams(max_tokens=3),
    )

    # 先让 A 完成 Prefill，再加入 B，构造 running 与 waiting 同时非空的策略分歧点。
    scheduler.add(seq_a)
    scheduled_seqs_output = schedule_legacy(scheduler)
    scheduled_seqs = scheduled_seqs_output.scheduled_seqs
    is_prefill = scheduled_seqs_output.is_prefill
    scheduler.postprocess(scheduled_seqs_output, token_ids=[100])
    scheduler.add(seq_b)

    # CPU 测试用固定 token 代替模型输出。token 只取决于 Sequence 和已完成数量，
    # 因此即使三种策略的执行顺序不同，最终输出也应该完全一致。
    num_steps = 0
    while not scheduler.is_finished():
        scheduled_seqs_output = schedule_legacy(scheduler)
        scheduled_seqs = scheduled_seqs_output.scheduled_seqs
        is_prefill = scheduled_seqs_output.is_prefill
        token_ids = [
            (100 if seq is seq_a else 200) + seq.num_completion_tokens
            for seq in scheduled_seqs
        ]
        scheduler.postprocess(scheduled_seqs_output, token_ids)
        num_steps += 1
        assert num_steps < 20  # 防止策略错误导致测试永久循环。

    assert seq_a.completion_token_ids == [100, 101, 102, 103]
    assert seq_b.completion_token_ids == [200, 201, 202]
    assert seq_a.status is SequenceStatus.FINISHED
    assert seq_b.status is SequenceStatus.FINISHED
    assert seq_a.num_cached_tokens == 0
    assert seq_b.num_cached_tokens == 0
    assert seq_a.block_table == []
    assert seq_b.block_table == []
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == []
    assert scheduler.block_manager.used_block_ids == set()
    assert len(scheduler.block_manager.free_block_ids) == config.num_kvcache_blocks
    assert all(block.ref_count == 0 for block in scheduler.block_manager.blocks)


# 场景：一个请求完成 Prefill 后位于 running，并持有一个物理 KV Block；随后它被选为 victim。
# 验证 preempt 将请求放回 waiting、清空其 KV 进度和 block_table，同时把物理 Block
# 从 used 集合归还到 free 队列，并将引用计数降为零；已生成的逻辑 token 仍然保留。
def test_preempt_moves_running_sequence_to_waiting_and_releases_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    """A preempted sequence releases its KV blocks and waits for prefill."""
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
        sampling_params=SamplingParams(max_tokens=3),
    )
    scheduler.add(seq)

    # Complete prefill so the request is running and owns a physical KV Block.
    prefill_seqs_output = schedule_legacy(scheduler)
    prefill_seqs = prefill_seqs_output.scheduled_seqs
    is_prefill = prefill_seqs_output.is_prefill
    scheduler.postprocess(prefill_seqs_output, token_ids=[90])

    allocated_block_ids = list(seq.block_table)
    assert seq.status is SequenceStatus.RUNNING
    assert seq.num_cached_tokens == 4
    assert seq.completion_token_ids == [90]
    assert scheduler.block_manager.used_block_ids == set(allocated_block_ids)
    assert all(
        scheduler.block_manager.blocks[block_id].ref_count == 1
        for block_id in allocated_block_ids
    )

    # Scheduler.preempt() expects its caller to remove the victim from running first.
    victim = scheduler.running.pop()
    assert victim is seq
    scheduler.preempt(victim)

    assert seq.status is SequenceStatus.WAITING
    assert seq.is_prefill is True
    assert seq.num_cached_tokens == 0
    assert seq.num_scheduled_tokens == 0
    assert seq.completion_token_ids == [90]
    assert list(scheduler.running) == []
    assert list(scheduler.waiting) == [seq]

    # 请求侧不再保存 logical block -> physical block 的映射。
    assert seq.block_table == []

    # 没有任何物理 Block 仍被标记为正在使用。
    assert scheduler.block_manager.used_block_ids == set()

    # 全部物理 Block 都已回到空闲队列；队列顺序不影响可用性。
    assert len(scheduler.block_manager.free_block_ids) == config.num_kvcache_blocks
    assert set(scheduler.block_manager.free_block_ids) == set(
        range(config.num_kvcache_blocks)
    )

    # 所有引用计数归零，说明没有 Sequence 继续持有这些物理 Block。
    assert all(block.ref_count == 0 for block in scheduler.block_manager.blocks)


# 场景：请求尚未达到 max_tokens=5，在一次 Decode 中返回 eos=99，且 ignore_eos=False。
# 验证 EOS 被记录后请求立即 FINISHED；结束原因确实是 EOS，而不是 max_tokens，
# 同时请求从 running 移除并释放其持有的全部 KV Block。
def test_eos_finishes_sequence_early_and_releases_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    """EOS finishes a sequence before max_tokens and releases its KV blocks."""
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
        sampling_params=SamplingParams(max_tokens=5, ignore_eos=False),
    )
    scheduler.add(seq)

    # Full prefill produces a normal first token, so the request keeps running.
    prefill_seqs_output = schedule_legacy(scheduler)
    prefill_seqs = prefill_seqs_output.scheduled_seqs
    is_prefill = prefill_seqs_output.is_prefill
    scheduler.postprocess(prefill_seqs_output, token_ids=[90])

    assert seq.completion_token_ids == [90]
    assert seq.status is SequenceStatus.RUNNING

    # The next decode returns EOS before the request reaches max_tokens.
    decode_seqs_output = schedule_legacy(scheduler)
    decode_seqs = decode_seqs_output.scheduled_seqs
    is_prefill = decode_seqs_output.is_prefill

    assert is_prefill is False
    assert decode_seqs == [seq]
    assert seq.num_scheduled_tokens == 1

    scheduler.postprocess(decode_seqs_output, token_ids=[99])

    assert seq.completion_token_ids == [90, 99]
    assert seq.num_completion_tokens == 2
    # 2 < 5 证明本次结束由 EOS 触发，而不是因为生成数量达到 max_tokens。
    assert seq.num_completion_tokens < seq.max_tokens
    assert seq.status is SequenceStatus.FINISHED
    assert seq.num_scheduled_tokens == 0
    assert seq.num_cached_tokens == 0
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == []
    assert scheduler.is_finished() is True

    # 请求结束后不再保存 logical block -> physical block 的映射。
    assert seq.block_table == []
    # 所有物理 Block 均解除占用并回到空闲池。
    assert scheduler.block_manager.used_block_ids == set()
    assert set(scheduler.block_manager.free_block_ids) == set(
        range(config.num_kvcache_blocks)
    )
    # 所有引用计数归零，避免请求结束后发生 KV Cache 泄漏。
    assert all(block.ref_count == 0 for block in scheduler.block_manager.blocks)


# 场景：与上一测试相同，但设置 ignore_eos=True；Decode 返回的 eos=99 应视为普通 token。
# 验证请求不会提前结束，仍留在 running，并继续持有已有 KV Block，直到以后达到 max_tokens。
def test_ignore_eos_keeps_sequence_running_and_blocks_allocated(
    monkeypatch: pytest.MonkeyPatch,
):
    """Ignored EOS is appended like a normal token without releasing KV blocks."""
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
        sampling_params=SamplingParams(max_tokens=5, ignore_eos=True),
    )
    scheduler.add(seq)

    prefill_seqs_output = schedule_legacy(scheduler)
    prefill_seqs = prefill_seqs_output.scheduled_seqs
    is_prefill = prefill_seqs_output.is_prefill
    scheduler.postprocess(prefill_seqs_output, token_ids=[90])

    decode_seqs_output = schedule_legacy(scheduler)
    decode_seqs = decode_seqs_output.scheduled_seqs
    is_prefill = decode_seqs_output.is_prefill
    assert is_prefill is False
    scheduler.postprocess(decode_seqs_output, token_ids=[99])

    assert seq.completion_token_ids == [90, 99]
    assert seq.num_completion_tokens == 2
    assert seq.num_completion_tokens < seq.max_tokens
    # ignore_eos=True 时，遇到 99 后请求仍需继续 Decode。
    assert seq.status is SequenceStatus.RUNNING
    assert list(scheduler.waiting) == []
    assert list(scheduler.running) == [seq]
    assert scheduler.is_finished() is False

    allocated_block_ids = list(seq.block_table)
    assert len(allocated_block_ids) == 2
    # 请求仍在运行，因此 logical -> physical Block 映射必须保留。
    assert seq.block_table == allocated_block_ids
    # 两个物理 Block 仍属于该请求，不能提前回收到 free 队列。
    assert scheduler.block_manager.used_block_ids == set(allocated_block_ids)
    assert len(scheduler.block_manager.free_block_ids) == (
        config.num_kvcache_blocks - len(allocated_block_ids)
    )
    # 每个被占用 Block 都恰好被当前 Sequence 引用一次。
    assert all(
        scheduler.block_manager.blocks[block_id].ref_count == 1
        for block_id in allocated_block_ids
    )
