import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


# 场景：4 个物理 Block 中，为一个 6-token、需要两个逻辑 Block 的请求分配缓存，再释放请求。
# 验证 allocate/deallocate 会同步维护 Sequence.block_table、used/free Block 集合和 ref_count，
# 最终所有物理 Block 都能完整回到空闲池，不留下资源泄漏或失效引用。
def test_allocate_and_deallocate_updates_block_state(
    monkeypatch: pytest.MonkeyPatch,
):
    """Exclusive allocation and release keep all block metadata consistent."""
    block_size = 4
    num_blocks = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)

    manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    seq = Sequence(
        token_ids=[10, 11, 12, 13, 14, 15],
        sampling_params=SamplingParams(max_tokens=2),
    )

    assert seq.num_blocks == 2
    assert manager.can_allocate(seq) == 0
    assert seq.block_table == []
    assert manager.used_block_ids == set()
    assert set(manager.free_block_ids) == set(range(num_blocks))

    manager.allocate(seq, num_cached_blocks=0)

    allocated_block_ids = list(seq.block_table)
    assert len(allocated_block_ids) == 2
    # Sequence 的两个逻辑 Block 均已映射到互不重复的物理 Block。
    assert len(set(allocated_block_ids)) == 2
    # used 集合必须与 Sequence 实际持有的物理 Block 完全一致。
    assert manager.used_block_ids == set(allocated_block_ids)
    # 其余两个 Block 仍在空闲池中，且不能与已分配 Block 重叠。
    assert set(manager.free_block_ids) == (
        set(range(num_blocks)) - set(allocated_block_ids)
    )
    # 当前只有这个 Sequence 使用这些 Block，所以每个引用计数都应为 1。
    assert all(
        manager.blocks[block_id].ref_count == 1
        for block_id in allocated_block_ids
    )

    manager.deallocate(seq)

    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    # 释放后没有物理 Block 仍被标记为占用。
    assert manager.used_block_ids == set()
    # 全部 Block ID 都回到空闲池；deque 中的先后顺序不影响可用性。
    assert set(manager.free_block_ids) == set(range(num_blocks))
    # 所有引用计数归零，说明没有 Sequence 遗留对物理 Block 的引用。
    assert all(block.ref_count == 0 for block in manager.blocks)


# 场景：两个 6-token 请求共享第一个完整的 4-token Prefix Block，但尾部 token 不同。
# 验证第二个请求命中同一个物理 Prefix Block 后 ref_count 变为 2；依次释放两个请求时，
# 共享 Block 先保持存活并服务另一个请求，直到最后一个引用解除后才回到空闲池。
def test_prefix_hit_shares_block_and_releases_on_last_reference(
    monkeypatch: pytest.MonkeyPatch,
):
    """A shared prefix block is freed only after its final owner releases it."""
    block_size = 4
    num_blocks = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)

    manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    seq1 = Sequence(
        token_ids=[10, 11, 12, 13, 20, 21],
        sampling_params=SamplingParams(max_tokens=2),
    )
    manager.allocate(seq1, num_cached_blocks=manager.can_allocate(seq1))

    # 模拟一次完整 Prefill：先登记新完成的完整 Block，再提交 cached_tokens 进度。
    seq1.num_scheduled_tokens = 6
    manager.hash_blocks(seq1)
    seq1.num_cached_tokens += seq1.num_scheduled_tokens
    seq1.num_scheduled_tokens = 0

    shared_block_id = seq1.block_table[0]
    seq1_private_block_id = seq1.block_table[1]
    prefix_hash = manager.blocks[shared_block_id].hash
    assert prefix_hash != -1
    assert manager.hash_to_block_id[prefix_hash] == shared_block_id

    seq2 = Sequence(
        token_ids=[10, 11, 12, 13, 30, 31],
        sampling_params=SamplingParams(max_tokens=2),
    )
    num_cached_blocks = manager.can_allocate(seq2)

    assert num_cached_blocks == 1
    manager.allocate(seq2, num_cached_blocks=num_cached_blocks)

    assert seq2.num_cached_tokens == block_size
    # 两个 Sequence 的第一个逻辑 Block 指向同一个物理 Prefix Block。
    assert seq1.block_table[0] == seq2.block_table[0] == shared_block_id
    # 共享 Block 同时被 seq1、seq2 引用，因此 ref_count 必须为 2。
    assert manager.blocks[shared_block_id].ref_count == 2

    manager.deallocate(seq1)

    assert seq1.block_table == []
    # seq1 的私有尾块无人继续使用，应立即回到空闲池。
    assert manager.blocks[seq1_private_block_id].ref_count == 0
    assert seq1_private_block_id in manager.free_block_ids
    # 共享 Block 仍被 seq2 引用，只能从 2 降到 1，不能提前释放。
    assert manager.blocks[shared_block_id].ref_count == 1
    assert shared_block_id in manager.used_block_ids
    assert shared_block_id not in manager.free_block_ids
    assert seq2.block_table[0] == shared_block_id

    manager.deallocate(seq2)

    # 最后一个引用解除后，共享 Prefix Block 才真正回到空闲池。
    assert manager.blocks[shared_block_id].ref_count == 0
    assert manager.used_block_ids == set()
    assert set(manager.free_block_ids) == set(range(num_blocks))
    assert all(block.ref_count == 0 for block in manager.blocks)


# 场景：一个带有效 Prefix hash 的空闲 Block 被另一个不同请求重新分配并准备覆盖。
# 验证重新分配该物理 Block 时会删除旧 hash_to_block_id 映射，并清空 Block 内旧的
# hash/token 元数据，防止后续请求把已经覆盖的 KV 数据误判成 Prefix Cache 命中。
def test_reallocating_cached_block_invalidates_old_hash(
    monkeypatch: pytest.MonkeyPatch,
):
    """Reusing a cached physical block invalidates its stale prefix index."""
    block_size = 4
    num_blocks = 2
    monkeypatch.setattr(Sequence, "block_size", block_size)

    manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
    old_seq = Sequence(
        token_ids=[10, 11, 12, 13, 20, 21],
        sampling_params=SamplingParams(max_tokens=2),
    )
    manager.allocate(old_seq, num_cached_blocks=manager.can_allocate(old_seq))

    old_seq.num_scheduled_tokens = 6
    manager.hash_blocks(old_seq)
    old_seq.num_cached_tokens += old_seq.num_scheduled_tokens
    old_seq.num_scheduled_tokens = 0

    old_prefix_block_id = old_seq.block_table[0]
    old_hash = manager.blocks[old_prefix_block_id].hash
    assert old_hash != -1
    assert manager.hash_to_block_id[old_hash] == old_prefix_block_id

    manager.deallocate(old_seq)

    # 仅释放时仍保留旧 hash，使空闲 Block 在未覆盖前仍可作为 Prefix Cache 命中。
    assert manager.blocks[old_prefix_block_id].hash == old_hash
    assert manager.hash_to_block_id[old_hash] == old_prefix_block_id
    assert old_prefix_block_id in manager.free_block_ids

    new_seq = Sequence(
        token_ids=[30, 31, 32, 33, 40, 41],
        sampling_params=SamplingParams(max_tokens=2),
    )
    assert manager.can_allocate(new_seq) == 0
    manager.allocate(new_seq, num_cached_blocks=0)

    # 两个 Block 都被新请求占用，旧 Prefix 所在物理 Block 必然已被重新分配。
    assert old_prefix_block_id in new_seq.block_table
    # 旧 hash 不能再指向即将存放新 K/V 的物理 Block。
    assert old_hash not in manager.hash_to_block_id
    assert manager.blocks[old_prefix_block_id].hash == -1
    assert manager.blocks[old_prefix_block_id].token_ids == []


# 场景：block_size=4，Prefill 后追加生成 token，使 Sequence 长度依次到达 5、6、7、8、9。
# 验证 allocate_slots 只在长度 5 和 9（新逻辑 Block 的第一个 token）分配物理 Block，
# 长度 6～8 仍复用当前尾块，避免少分配导致越界或每个 token 都错误分配新 Block。
def test_allocate_slots_allocates_only_at_block_boundaries(
    monkeypatch: pytest.MonkeyPatch,
):
    """Decode allocates a physical block only when entering a new logical block."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)

    manager = BlockManager(num_blocks=4, block_size=block_size)
    seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=8),
    )
    manager.allocate(seq, num_cached_blocks=manager.can_allocate(seq))
    assert len(seq.block_table) == 1

    # 模拟完整 Prefill 已经写入 4 个 KV token；随后追加首个生成 token。
    # 下一轮 Decode 前，新增 token 位于第二个逻辑 Block，需要分配第二个物理 Block。
    seq.num_cached_tokens = 4
    seq.append_token(14)
    assert manager.can_allocate_slots(seq) is True
    manager.allocate_slots(seq)
    assert len(seq.block_table) == 2
    assert len(seq.block_table) == seq.num_blocks

    # 长度 6、7、8 都仍位于第二个逻辑 Block，不应重复分配。
    seq.num_cached_tokens += 1
    seq.append_token(15)
    manager.allocate_slots(seq)
    assert len(seq.block_table) == 2
    seq.num_cached_tokens += 1
    seq.append_token(16)
    manager.allocate_slots(seq)
    assert len(seq.block_table) == 2
    seq.num_cached_tokens += 1
    seq.append_token(17)
    manager.allocate_slots(seq)
    assert len(seq.block_table) == 2

    # 长度 9 跨入第三个逻辑 Block，因此再次分配一个物理 Block。
    seq.num_cached_tokens += 1
    seq.append_token(18)
    assert manager.can_allocate_slots(seq) is True
    manager.allocate_slots(seq)
    assert len(seq.block_table) == 3
    assert len(seq.block_table) == seq.num_blocks
    assert manager.used_block_ids == set(seq.block_table)


# 场景：Sequence 已完成前 4 个 token 的计算并持有一个物理 Block，本轮一次需要计算
# 后续 8 个 token。验证 allocate_slots 会根据本轮结束位置一次性补齐两个物理 Block，
# 而不是只处理一个 token 或每个 token 重复申请 Block。
def test_allocate_slots_supports_multiple_new_tokens(
    monkeypatch: pytest.MonkeyPatch,
):
    """Allocating eight new tokens can add multiple physical blocks at once."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)

    manager = BlockManager(num_blocks=4, block_size=block_size)
    seq = Sequence(
        token_ids=[10, 11, 12, 13],
        sampling_params=SamplingParams(max_tokens=8),
    )
    manager.allocate(seq, num_cached_blocks=manager.can_allocate(seq))
    seq.num_cached_tokens = 4

    assert len(seq.block_table) == 1
    assert manager._num_required_new_blocks(seq, num_new_tokens=8) == 2
    assert manager.can_allocate_slots(seq, num_new_tokens=8) is True

    manager.allocate_slots(seq, num_new_tokens=8)

    assert len(seq.block_table) == 3
    assert len(manager.used_block_ids) == 3
    assert manager.used_block_ids == set(seq.block_table)


# 场景：Chunked Prefill 已经为完整 Prompt 提前分配了全部物理 Block，但本轮只计算其中
# 一小段 token。验证新增物理 Block 数量为 0，allocate_slots 不会重复分配。
def test_allocate_slots_reuses_preallocated_chunked_prefill_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    """Preallocated chunked-prefill blocks are not allocated a second time."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)

    manager = BlockManager(num_blocks=4, block_size=block_size)
    seq = Sequence(
        token_ids=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        sampling_params=SamplingParams(max_tokens=2),
    )
    manager.allocate(seq, num_cached_blocks=manager.can_allocate(seq))
    initial_block_table = list(seq.block_table)

    assert len(initial_block_table) == 3
    assert seq.num_cached_tokens == 0
    assert manager._num_required_new_blocks(seq, num_new_tokens=4) == 0
    assert manager.can_allocate_slots(seq, num_new_tokens=4) is True

    manager.allocate_slots(seq, num_new_tokens=4)

    assert seq.block_table == initial_block_table
    assert manager.used_block_ids == set(initial_block_table)


# 场景：BlockManager 只有一个物理 Block，但 6-token 请求需要两个逻辑 Block。
# 验证 can_allocate 返回 -1 表示容量不足，并且检查过程本身不做部分分配，
# Sequence 和 BlockManager 的 block_table、used/free 集合、ref_count 均保持原状。
def test_can_allocate_returns_minus_one_without_changing_state(
    monkeypatch: pytest.MonkeyPatch,
):
    """An allocation capacity check fails without mutating block state."""
    block_size = 4
    monkeypatch.setattr(Sequence, "block_size", block_size)

    manager = BlockManager(num_blocks=1, block_size=block_size)
    seq = Sequence(
        token_ids=[10, 11, 12, 13, 14, 15],
        sampling_params=SamplingParams(max_tokens=2),
    )

    assert seq.num_blocks == 2
    assert manager.can_allocate(seq) == -1
    # 容量检查失败后，请求不能得到任何部分分配的物理 Block。
    assert seq.block_table == []
    assert seq.num_cached_tokens == 0
    # 唯一的物理 Block 仍完整地留在空闲池中。
    assert manager.used_block_ids == set()
    assert list(manager.free_block_ids) == [0]
    assert manager.blocks[0].ref_count == 0
