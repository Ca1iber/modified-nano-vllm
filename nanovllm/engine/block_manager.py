from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

# 元数据
class Block:

    # 初始化一个物理 KV-cache block 的 CPU 侧元数据，此时还没有 Sequence 使用它。
    def __init__(self, block_id):
        self.block_id = block_id
        # 物理 KV block 的引用计数，表示当前有多少个 Sequence 的 block_table 正在使用这个物理 block
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    # 记录完整 token block 的链式 hash 和 token 内容，供 prefix cache 后续查找与校验。
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    # 将 block 标记为已被一个请求占用，并清除即将被覆盖的旧 prefix-cache 元数据。
    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    # 建立固定大小的物理 block 池，并维护空闲、占用和 prefix-cache 三类索引。
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()      # hash and block_id
        # 创建一个元素为 [0, num_blocks] 的双端队列，用来初始化表示所有的 block 都是空闲的
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 创建一个集合，元素不能重复，不保证固定顺序，可以快速判断某个元素是否存在
        self.used_block_ids: set[int] = set()

    # 将前一个 block 的 hash 与当前 token block 一起压缩成 64-bit 链式 hash。
    # 普通方法自动接收“具体对象” self；@classmethod 自动接收“类本身” cls。
    # 这个方法的行为不依赖某个具体对象，但可能需要访问类级别的信息。
    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # 从空闲池取出一个物理 block；若其中保留着旧 prefix cache，先让旧索引失效。
    # 这里只分配编号并更新元数据，真正的 GPU K/V 会在 Attention 中按 slot_mapping 写入。
    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()    # 取空闲 blk id
        block = self.blocks[block_id]               # 取该 blk 数据 
        assert block.ref_count == 0
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]   # 删除存着的旧 K/V Cache 和 hash
        block.reset()                               # 重置元数据
        self.used_block_ids.add(block_id)           # 加入正在使用的集合
        return block_id                             # 上层 allocate() 会将这个编号放进 Sequence 的 block_table

    # 将引用计数已归零的 block 放回空闲池，但不清零 GPU K/V，也不立即删除有效 hash。
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # 计算当前序列 Prefix Cache 命中了多少个 blocks
    # 解决原 can_allocate 中可能出现的分配冗余问题：默认为整个 prompt
    # 计算所需 block 数，但是实际上有可能因为 budget 原因没法分配那么多
    # 会导致浪费
    def get_num_cached_blocks(self, seq: Sequence) -> int:
        h = -1
        num_cached_blocks = 0

        # 最后一个逻辑 block 不参与命中，保证至少重新计算最后部分以获得 logits
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)

            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1:
                break

            block = self.blocks[block_id]
            if block.token_ids != token_ids:
                break

            num_cached_blocks += 1

        return num_cached_blocks

    # 判断剩余物理 block 是否足够 num_new_tokens
    def can_allocate(self, seq: Sequence, num_new_tokens: int | None = None) -> bool:
        num_cached_blocks = self.get_num_cached_blocks(seq)

        # legacy 检查完整 Sequence
        if num_new_tokens is None:
            end_token = seq.num_tokens
        else:
            end_token = num_cached_blocks * self.block_size + num_new_tokens

        num_required_blocks = (end_token - 1 + self.block_size) // self.block_size
        num_required_free_blocks = num_required_blocks
        # 重新沿 Prefix hash 遍历已经命中的 blocks
        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)

            if block_id in self.used_block_ids:
                num_required_free_blocks -= 1

        if len(self.free_block_ids) < num_required_free_blocks:
            return False

        return True

    # 为新 Sequence 填充 block_table：先复用命中的 prefix blocks，再分配未命中的 blocks。
    def allocate(self, seq: Sequence, num_new_tokens: int | None = None):
        assert not seq.block_table

        num_cached_blocks = self.get_num_cached_blocks(seq)

        if num_new_tokens is None:
            num_required_blocks = seq.num_blocks
        else:
            end_token = num_cached_blocks * self.block_size + num_new_tokens
            num_required_blocks = (end_token - 1 + self.block_size) // self.block_size

        h = -1
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)            # 取出第 i 个 cached_block 的 tokens
            h = self.compute_hash(token_ids, h) # 计算 hash
            block_id = self.hash_to_block_id[h] # 查字典看这个 cached_block 的 block_id
            block = self.blocks[block_id]       # 拿对应 block_id 的元数据
            if block_id in self.used_block_ids: # 该 cached_block 正在被其他 Sequence 使用
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for _ in range(num_cached_blocks, num_required_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    # 释放 Sequence 对所有物理 blocks 的引用；引用归零的 block 才回到空闲池。
    # 同时清空该 Sequence 的 block_table 和已缓存 token 计数。
    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    # 检查是否剩余足够多的 kv blocks 可以分配
    # 新增 num_new_tokens 且默认为 1 以兼容 Legacy
    def can_allocate_slots(self, seq: Sequence, num_new_tokens: int = 1) -> bool:
        num_required_new_blocks = self._num_required_new_blocks(seq, num_new_tokens)
        return len(self.free_block_ids) >= num_required_new_blocks

    # 分配 kv blocks
    # 新增 num_new_tokens 且默认为 1 以兼容 Legacy
    def allocate_slots(self, seq: Sequence, num_new_tokens: int = 1):
        num_required_new_blocks = self._num_required_new_blocks(seq, num_new_tokens)
        for _ in range(num_required_new_blocks):
            seq.block_table.append(self._allocate_block())

    # 找出本轮新计算完整的 token blocks，为其生成链式 hash 并写入 Block 元数据。
    # 同时登记 hash -> physical block_id，供后续请求查询 prefix cache。
    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id

    # 本轮结束位置所需要覆盖的总逻辑 block 数
    def _num_all_blocks(self, seq: Sequence, num_new_tokens: int) -> int:
        end_token = seq.num_cached_tokens + num_new_tokens
        return (end_token - 1 + self.block_size) // self.block_size

    # 计算本轮新覆盖的逻辑 block 数
    def _num_new_logical_blocks(self, seq: Sequence, num_new_tokens: int) -> int:
        num_current_blocks = (
            seq.num_cached_tokens - 1 + self.block_size) // self.block_size
        return self._num_all_blocks(seq, num_new_tokens) - num_current_blocks

    # 计算本轮需要新增的物理 block 数
    def _num_required_new_blocks(self, seq: Sequence, num_new_tokens: int) -> int:
        return max(self._num_all_blocks(seq, num_new_tokens) - len(seq.block_table), 0)
