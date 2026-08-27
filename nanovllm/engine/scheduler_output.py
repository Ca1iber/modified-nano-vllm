from dataclasses import dataclass
from nanovllm.engine.sequence import Sequence
from abc import ABC

@dataclass(slots=True)
class SchedulerOutput(ABC):
    scheduled_seqs: list[Sequence]
    # key: seq_id, value: 本 Step 实际调度的 token 数量；注意可能小于请求的 max_tokens。
    # num_scheduled_tokens: dict[int, int]
    total_num_scheduled_tokens: int

    # should_sample：P1.2c 的采样边界


@dataclass(slots=True)
class LegacySchedulerOutput(SchedulerOutput):
    # 当前整个 batch 共用一个 is_prefill
    is_prefill: bool


@dataclass(slots=True)
class UnifiedSchedulerOutput(SchedulerOutput):
    # 表示本轮哪些位置需要拿出来采样生成新 token 
    should_sample: list[bool]
    # 可以表达 prefill 和 decode 的混合 batch
    is_prefilling: list[bool]