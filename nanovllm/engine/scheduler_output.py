from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from nanovllm.engine.sequence import Sequence


@dataclass(slots=True)
class SchedulerOutput(ABC):
    scheduled_seqs: list[Sequence]
    # key: seq_id, value: 本 Step 实际调度的 token 数量；注意可能小于请求的 max_tokens。
    # num_scheduled_tokens: dict[int, int]
    total_num_scheduled_tokens: int

    @abstractmethod
    def to_unified(self) -> UnifiedSchedulerOutput:
        ...


@dataclass(slots=True)
class LegacySchedulerOutput(SchedulerOutput):
    # 当前整个 batch 共用一个 is_prefill
    is_prefill: bool

    def to_unified(self) -> UnifiedSchedulerOutput:
        scheduled_seqs = self.scheduled_seqs
        total_num_scheduled_tokens = self.total_num_scheduled_tokens
        is_prefilling = [self.is_prefill] * len(scheduled_seqs)
        should_sample = [
            seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens
            for seq in scheduled_seqs
        ]

        return UnifiedSchedulerOutput(
            scheduled_seqs=scheduled_seqs,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            should_sample=should_sample,
            is_prefilling=is_prefilling,
        )


@dataclass(slots=True)
class UnifiedSchedulerOutput(SchedulerOutput):
    # 表示本轮哪些位置需要拿出来采样生成新 token 
    should_sample: list[bool]
    # 可以表达 prefill 和 decode 的混合 batch
    is_prefilling: list[bool]

    def to_unified(self) -> UnifiedSchedulerOutput:
        return self
