import os

import pytest
import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.attention import store_kvcache
from nanovllm.utils.context import get_context, reset_context


def require_cuda_gpu():
    """只在用户显式开启且 CUDA 可用时运行底层 GPU 正确性测试。"""
    if os.environ.get("NANOVLLM_RUN_GPU_TESTS") != "1":
        pytest.skip("设置 NANOVLLM_RUN_GPU_TESTS=1 后才运行 GPU 正确性测试")
    if not torch.cuda.is_available():
        pytest.skip("当前环境没有可用 CUDA GPU")


@pytest.fixture(autouse=True)
def clean_attention_context():
    """避免当前测试写入的全局 Attention Context 影响后续测试。"""
    reset_context()
    yield
    reset_context()


def make_metadata_runner(block_size: int) -> ModelRunner:
    """创建只调用 metadata 准备方法、不会加载模型或初始化 NCCL 的 ModelRunner。"""
    runner = object.__new__(ModelRunner)
    runner.block_size = block_size
    return runner


# 场景：block_size=4，Sequence 的逻辑 Block 0/1 分别映射到不连续的物理
# Block 7/2；位置 0、1 已有 KV，本轮 Chunked Prefill 继续处理位置 2～5。
# 验证扁平输入与 position 都是 2、3、4、5，并且 slot_mapping 先写完物理
# Block 7 的 slot 30、31，再跳到物理 Block 2 的 slot 8、9；这可以发现从
# Block 中间恢复、跨边界切换物理 Block 以及边界处 +1/-1 的地址计算错误。
@pytest.mark.gpu
def test_prefill_slot_mapping_crosses_noncontiguous_block_boundary():
    require_cuda_gpu()
    runner = make_metadata_runner(block_size=4)
    seq = Sequence([10, 11, 12, 13, 14, 15])
    seq.block_size = 4
    seq.block_table = [7, 2]
    seq.num_cached_tokens = 2
    seq.num_scheduled_tokens = 4

    input_ids, positions = runner.prepare_prefill([seq])
    context = get_context()

    assert input_ids.cpu().tolist() == [12, 13, 14, 15]
    assert positions.cpu().tolist() == [2, 3, 4, 5]
    assert context.slot_mapping.cpu().tolist() == [30, 31, 8, 9]
    assert context.cu_seqlens_q.cpu().tolist() == [0, 4]
    assert context.cu_seqlens_k.cpu().tolist() == [0, 6]
    assert context.block_tables.cpu().tolist() == [[7, 2]]


# 场景：block_size=4，一条 Sequence 已从 4 个 token 增长到 5 个 token，因而
# 最新 token 位于逻辑 Block 1 的第一个位置；该逻辑 Block 映射到物理 Block 2。
# 验证 Decode 只输入位置 4 的最后一个 token，并把它的 K/V 写入 slot 8，而不是
# 假设物理 Block 连续后误写到 slot 32；Prefill 和 Decode 使用不同的 metadata
# 准备代码，所以这里单独固定 Decode 刚跨入新 Block 时的地址计算行为。
@pytest.mark.gpu
def test_decode_slot_mapping_uses_first_slot_of_new_physical_block():
    require_cuda_gpu()
    runner = make_metadata_runner(block_size=4)
    seq = Sequence([10, 11, 12, 13, 14])
    seq.block_size = 4
    seq.block_table = [7, 2]
    seq.is_prefill = False

    input_ids, positions = runner.prepare_decode([seq])
    context = get_context()

    assert input_ids.cpu().tolist() == [14]
    assert positions.cpu().tolist() == [4]
    assert context.slot_mapping.cpu().tolist() == [8]
    assert context.context_lens.cpu().tolist() == [5]
    assert context.block_tables.cpu().tolist() == [[7, 2]]


# 场景：为四个 token 构造彼此不同且可精确比较的 K/V，并使用跨物理 Block 的
# slot_mapping=[30, 31, 8, 9] 调用真实 Triton store_kvcache Kernel。
# 验证四份数据分别出现在指定 slot，同时把整个期望 Cache 的其他位置保持为 0；
# 这不仅检查“地址表算出来了”，还检查 Kernel 是否按 token 下标读取对应地址，
# 并且没有在 Block 边界处漏写、错写或破坏任何相邻 slot。
@pytest.mark.gpu
def test_store_kvcache_writes_only_mapped_slots_across_blocks():
    require_cuda_gpu()
    num_blocks = 8
    block_size = 4
    num_heads = 2
    head_dim = 4
    target_slots = [30, 31, 8, 9]

    key = torch.arange(1, 33, dtype=torch.float32, device="cuda").reshape(4, num_heads, head_dim)
    value = key + 100
    k_cache = torch.zeros(num_blocks, block_size, num_heads, head_dim, device="cuda")
    v_cache = torch.zeros_like(k_cache)
    slot_mapping = torch.tensor(target_slots, dtype=torch.int32, device="cuda")

    store_kvcache(key, value, k_cache, v_cache, slot_mapping)

    expected_k_cache = torch.zeros_like(k_cache).view(-1, num_heads, head_dim)
    expected_v_cache = torch.zeros_like(v_cache).view(-1, num_heads, head_dim)
    expected_k_cache[target_slots] = key
    expected_v_cache[target_slots] = value

    assert torch.equal(k_cache.view(-1, num_heads, head_dim), expected_k_cache)
    assert torch.equal(v_cache.view(-1, num_heads, head_dim), expected_v_cache)
