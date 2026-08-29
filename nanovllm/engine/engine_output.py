from dataclasses import dataclass


# 某个请求在本轮采样出的一个 token
@dataclass(slots=True)
class SampledTokenOutput:
    seq_id: int
    token_id: int


# 某个请求完成时的完整 completion token
@dataclass(slots=True)
class FinishedRequestOutput:
    seq_id: int
    final_completion_token_ids: list[int]


# EngineStepOutput 是 LLMEngine.step() 对上层返回的统一结果
# 用来替代现在这个语义隐晦的二元组：outputs, signed_num_tokens
@dataclass(slots=True)
class EngineStepOutput:
    # 本轮刚刚完成的请求
    finished_requests: list[FinishedRequestOutput]

    # 本轮新采样出的 token
    sampled_tokens: list[SampledTokenOutput]

    num_step_prefill_tokens: int
    num_step_decode_tokens: int

    @property
    def total_num_tokens(self) -> int:
        return self.num_step_decode_tokens + self.num_step_prefill_tokens
