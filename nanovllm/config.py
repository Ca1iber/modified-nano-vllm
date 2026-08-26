import os
from dataclasses import dataclass
from transformers import AutoConfig

SCHEDULER_POLICIES = ("prefill_first", "decode_first", "time_sliced")


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    # 默认关闭请求计时，避免统计功能给原有推理路径增加不必要的开销。
    enable_stats: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # 调度策略：默认保持原有的 Prefill-first 行为。
    scheduler_policy: str = "prefill_first"
    # time_sliced 策略连续执行 Decode 的最多 Step 数。
    time_sliced_decode_steps: int = 4
    # 选择新旧调度模式
    scheduler_mode: str = "legacy"

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        if self.num_kvcache_blocks != -1 and self.num_kvcache_blocks <= 0:
            raise ValueError("num_kvcache_blocks 必须为 -1（自动）或正整数")
        if self.scheduler_policy not in SCHEDULER_POLICIES:
            raise ValueError(
                f"scheduler_policy 必须是 {SCHEDULER_POLICIES} 之一"
            )
        if self.time_sliced_decode_steps <= 0:
            raise ValueError("time_sliced_decode_steps 必须为正整数")
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
