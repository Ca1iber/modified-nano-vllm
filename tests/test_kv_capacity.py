import pytest

from nanovllm.engine.model_runner import resolve_num_kvcache_blocks


# 场景：用户不显式限制 KV Cache，配置值保持默认 -1。验证 -1 只是“自动计算”的
# 哨兵值，最终会解析成当前 GPU 安全容量，而不是真的尝试分配负数个 Block。
def test_automatic_kv_capacity_uses_all_available_blocks():
    assert resolve_num_kvcache_blocks(-1, available_blocks=12) == 12


# 场景：GPU 明明可以安全容纳 12 个 Block，但抢占实验只要求使用 2 个。验证显式
# 正整数会原样保留，不再被 ModelRunner 的自动显存计算结果覆盖。
def test_explicit_kv_capacity_is_not_overwritten():
    assert resolve_num_kvcache_blocks(2, available_blocks=12) == 2


# 场景：除 -1 外，0 和其他负数既不表示自动，也不是有效容量。验证这些输入会在
# 分配前得到清楚错误，避免创建空 BlockManager 或把非法数量传给 torch.empty。
@pytest.mark.parametrize("requested_blocks", [0, -2])
def test_kv_capacity_rejects_non_positive_values_other_than_auto(requested_blocks):
    with pytest.raises(ValueError, match="-1（自动）或正整数"):
        resolve_num_kvcache_blocks(requested_blocks, available_blocks=12)


# 场景：用户显式请求的 Block 数超过 gpu_memory_utilization 允许的安全容量。验证
# 程序直接报告请求值和上限，而不是继续分配并最终抛出难理解的 CUDA OOM。
def test_explicit_kv_capacity_cannot_exceed_available_blocks():
    with pytest.raises(ValueError, match="请求 13.*最多只能安全分配 12"):
        resolve_num_kvcache_blocks(13, available_blocks=12)


# 场景：模型、CUDA Graph 和其他运行内存已经占满显存，连一个 KV Block 都放不下。
# 自动与显式模式都必须在创建 KV Tensor 前失败，不能留下容量为零的 Scheduler。
@pytest.mark.parametrize("requested_blocks", [-1, 1])
def test_kv_capacity_rejects_gpu_with_no_available_block(requested_blocks):
    with pytest.raises(ValueError, match="不足以分配一个"):
        resolve_num_kvcache_blocks(requested_blocks, available_blocks=0)
