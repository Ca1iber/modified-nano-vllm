from collections import deque

from nanovllm.config import Config, SCHEDULER_POLICIES
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.stats import EngineStats
from nanovllm.engine.scheduler_output import (LegacySchedulerOutput,
                                              UnifiedSchedulerOutput,
                                              SchedulerOutput)

class Scheduler:

    def __init__(self, config: Config, stats: EngineStats | None = None):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.scheduler_policy = getattr(config, "scheduler_policy", "prefill_first")
        self.time_sliced_decode_steps = getattr(config, "time_sliced_decode_steps", 4)
        if self.scheduler_policy not in SCHEDULER_POLICIES:
            raise ValueError(f"scheduler_policy 必须是 {SCHEDULER_POLICIES} 之一")
        if self.time_sliced_decode_steps <= 0:
            raise ValueError("time_sliced_decode_steps 必须为正整数")
        # time_sliced 用它记录上一次 Prefill 后已经连续执行了多少个 Decode Step。
        self.decode_steps_since_prefill = 0
        # stats=None 时所有调度和状态转换都沿用原来的路径，不产生计时开销。
        self.stats = stats
        # 创建一个装 Sequence 类型的双端队列，可高效从左右两端添加和删除
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        # 确认调度模式(getattr写法兼容旧测试)
        self.scheduler_mode = getattr(config, 'scheduler_mode', 'legacy')

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
        if self.stats is not None:
            # arrival 的口径是“成功进入 Scheduler.waiting”，不包含 tokenizer 时间。
            self.stats.record_arrival(seq.seq_id)

    def schedule(self) -> LegacySchedulerOutput | UnifiedSchedulerOutput:
        if self.scheduler_mode == "legacy":
            return self._schedule_legacy()

        if self.scheduler_mode == "unified":
            return self._schedule_unified()

        raise ValueError(
            f"不支持的 scheduler_mode: {self.scheduler_mode}"
        )

    def _schedule_legacy(self) -> LegacySchedulerOutput:
        # 根据策略选择本 Step 先尝试 Prefill 还是 Decode。
        for is_prefill, phase in self._phase_order_legacy():
            scheduled_seqs = phase()
            if scheduled_seqs:
                self._record_scheduled(scheduled_seqs)
                if is_prefill:
                    self.decode_steps_since_prefill = 0
                else:
                    self.decode_steps_since_prefill += 1
                return LegacySchedulerOutput(
                    scheduled_seqs=scheduled_seqs,
                    # num_scheduled_tokens={
                    #     seq.seq_id: seq.num_scheduled_tokens
                    #     for seq in scheduled_seqs
                    # },
                    total_num_scheduled_tokens=sum(
                        seq.num_scheduled_tokens for seq in scheduled_seqs
                    ),
                    is_prefill=is_prefill,
                )
        raise RuntimeError("Scheduler 没有可调度的请求")

    def _schedule_unified(self) -> UnifiedSchedulerOutput:
        # scheduled_seqs[i], is_prefilling[i], should_sample[i] 三者按位置一一对应
        scheduled_seqs: list[Sequence] = []
        # 用来表征某个 Sequence 是在 prefill 还是在 decode
        is_prefilling: list[bool] = []
        # 用来区分某个 Sequence 是否需要采样出新 token
        should_sample: list[bool] = []

        token_budget = self.max_num_batched_tokens
        # 构造本轮两种请求的快照
        running_seqs = list(self.running)
        waiting_seqs = list(self.waiting)
        
        preemption_happened: bool = False

        # 先处理 running 请求
        for seq in running_seqs:
            if seq.status is not SequenceStatus.RUNNING:
                continue

            if len(scheduled_seqs) >= self.max_num_seqs:
                break

            if token_budget == 0:
                break

            num_new_tokens = seq.num_tokens - seq.num_cached_tokens
            num_new_tokens = min(num_new_tokens, token_budget)

            if num_new_tokens == 0:
                continue

            # kv cache 不够，抢占！
            while not self.block_manager.can_allocate_slots(seq, num_new_tokens):
                if self.running:
                    victim = self.running.pop()
                else:
                    victim = seq

                # 放回 waiting 队列的操作在此完成
                self.preempt(victim)
                preemption_happened = True

                if victim is seq:
                    break

            # 防止抢占自己之后继续执行后续 decode 逻辑
            if seq.status is not SequenceStatus.RUNNING:
                continue

            # 1. 分配真正的 kv cache
            self.block_manager.allocate_slots(seq, num_new_tokens)
            # 2. 保存本轮实际调度的 token 数
            seq.num_scheduled_tokens = num_new_tokens
            # 3. 将请求和对应 metadata 加入列表
            scheduled_seqs.append(seq)
            is_prefilling.append(False)
            should_sample.append(
                seq.num_cached_tokens + num_new_tokens == seq.num_tokens
            )
            # 4.最后扣除预算
            token_budget -= num_new_tokens

        # 再处理 waiting 请求
        if not preemption_happened:
            for seq in waiting_seqs:
                if seq.status is not SequenceStatus.WAITING:
                    continue

                if len(scheduled_seqs) >= self.max_num_seqs:
                    break

                if token_budget == 0:
                    break


        raise NotImplementedError("Unified Scheduler 尚未实现")

    def _phase_order_legacy(self):
        """返回本轮的候选阶段；阶段函数只负责尝试调度，不负责计时或计数。"""
        if self.scheduler_policy == "prefill_first":
            return ((True, self._schedule_prefill_legacy), (False, self._schedule_decode_legacy))
        if self.scheduler_policy == "decode_first":
            return ((False, self._schedule_decode_legacy), (True, self._schedule_prefill_legacy))
        # time_sliced：达到 Decode 配额且确实有等待请求时，优先给等待请求一次机会。
        if (
            self.waiting
            and self.running
            and self.decode_steps_since_prefill >= self.time_sliced_decode_steps
        ):
            return ((True, self._schedule_prefill_legacy), (False, self._schedule_decode_legacy))
        return ((False, self._schedule_decode_legacy), (True, self._schedule_prefill_legacy))

    def _record_scheduled(self, scheduled_seqs: list[Sequence]):
        """把真正进入本 Step 的请求登记到统计对象；关闭统计时不做任何操作。"""
        if self.stats is not None:
            self.stats.record_scheduled([seq.seq_id for seq in scheduled_seqs])

    def _schedule_prefill_legacy(self) -> list[Sequence]:
        # scheduled_seqs = 当前 Prefill Step 具体让谁上 GPU
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:   # block_table 不为空，上次 chunked prefill 没做完走的分支
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
                if self.stats is not None:
                    self.stats.record_prefix_cache_hit(
                        seq.seq_id,
                        num_blocks=num_cached_blocks,
                        block_size=self.block_size,
                    )
            scheduled_start = seq.num_cached_tokens
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if self.stats is not None:
                # 普通 Prefill 的历史计算边界为 0，因此不会被误记为 Recompute。
                self.stats.record_recompute(
                    seq.seq_id,
                    scheduled_start=scheduled_start,
                    num_scheduled_tokens=seq.num_scheduled_tokens,
                )
            # 判断本次 prefill 能否全部完成
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs

        return []

    def _schedule_decode_legacy(self) -> list[Sequence]:
        # Decode Step 中，每个被选中的 running 请求只计算一个 token。
        scheduled_seqs = []
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            # 当前 Sequence 没有足够的 KV Cache 空间容纳这次 Decode token，因此需要先释放空间
            while not self.block_manager.can_allocate_slots(seq):
                if self.running:    # running 中还有其他 seq
                    self.preempt(self.running.pop())    # 抢占其他请求
                else:
                    self.preempt(seq)                   # 否则抢占自己
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                # 如果新 token 正好跨入一个新逻辑 block，就分配一个新的物理 KV block
                self.block_manager.allocate_slots(seq)
                scheduled_seqs.append(seq)
        if scheduled_seqs:
            self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        if self.stats is not None:
            cached_tokens = seq.num_cached_tokens
            released_block_references = len(seq.block_table)
            num_free_blocks_before = len(self.block_manager.free_block_ids)
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        if self.stats is not None:
            # 共享 Block 只降低 ref_count；只有进入 free 队列的才算真正释放。
            freed_physical_blocks = (
                len(self.block_manager.free_block_ids) - num_free_blocks_before
            )
            self.stats.record_preemption(
                seq.seq_id,
                cached_tokens=cached_tokens,
                released_block_references=released_block_references,
                freed_physical_blocks=freed_physical_blocks,
            )

    # ModelRunner 完成本轮计算后，把结果写回各个 Sequence，并更新调度状态。
    '''
        1. 配对 Sequence 和采样 token
        2. 登记新完成的 Prefix Cache block
        3. 更新 Prefill/Decode 进度
        4. Chunked Prefill 未完成就跳过采样结果
        5. 追加新生成的 token
        6. 判断请求是否结束
    '''
    def postprocess(self, scheduler_output: SchedulerOutput, token_ids: list[int]):
        if isinstance(scheduler_output, LegacySchedulerOutput):
            self._postprocess_legacy(scheduler_output, token_ids)

        elif isinstance(scheduler_output, UnifiedSchedulerOutput):
            self._postprocess_unified(scheduler_output, token_ids)

        else:
            raise TypeError(
                f"不支持的 SchedulerOutput 类型: "
                f"{type(scheduler_output).__name__}"
            )

    def _postprocess_legacy(self, scheduler_output: LegacySchedulerOutput, token_ids: list[int]):
        for seq, token_id in zip(scheduler_output.scheduled_seqs, token_ids):
            # 后期接口更改?
            self.block_manager.hash_blocks(seq)

            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0

            # 还没完成 prefill
            if (scheduler_output.is_prefill and seq.num_cached_tokens < seq.num_tokens):
                continue

            seq.append_token(token_id)

            finished = (
                (not seq.ignore_eos and token_id == self.eos)
                or seq.num_completion_tokens == seq.max_tokens
            )
            if finished:
                seq.status = SequenceStatus.FINISHED

            if self.stats is not None:
                self.stats.record_output_token(seq.seq_id, finished=finished)

            if finished:
                self.block_manager.deallocate(seq)
                self.running.remove(seq)


    def _postprocess_unified(self, scheduler_output: UnifiedSchedulerOutput, token_ids: list[int]):
        raise NotImplementedError(
            '_postprocess_unified 尚未实现'
        )