"""为 nano-vLLM benchmark 构造固定结构、可由 seed 复现的请求集合。"""

from dataclasses import dataclass, field
from random import Random


DEFAULT_WORKLOAD_NAME = "random_mixed"
TOKEN_ID_MIN = 0
TOKEN_ID_MAX = 10000
# P0.4 正式验收的六张试卷；random_mixed 只保留原 bench.py 的兼容入口。
OFFICIAL_WORKLOAD_NAMES = (
    "short_prompt_long_decode",
    "long_prompt_short_decode",
    "mixed_lengths",
    "shared_prefix_high_hit",
    "kv_pressure_preemption",
    "decode_then_long_prefill",
)


@dataclass(slots=True)
class RequestSpec:
    """一个 benchmark 请求的输入、输出长度、分组和到达 Step。"""

    prompt_token_ids: list[int]
    max_tokens: int
    group: str
    arrival_step: int = 0

    def __post_init__(self):
        if not self.prompt_token_ids:
            raise ValueError("benchmark Prompt 不能为空")
        if self.max_tokens <= 0:
            raise ValueError("benchmark max_tokens 必须大于 0")
        if not self.group:
            raise ValueError("benchmark 请求必须属于一个非空分组")
        if self.arrival_step < 0:
            raise ValueError("benchmark arrival_step 不能小于 0")


@dataclass(slots=True)
class WorkloadSpec:
    """一张固定 benchmark 试卷及其运行所需的引擎容量配置。"""

    name: str
    description: str
    requests: list[RequestSpec]
    max_model_len: int
    max_num_batched_tokens: int
    max_num_seqs: int
    # -1 表示按 GPU 显存自动计算；正整数用于固定 KV 压力、复现实验。
    num_kvcache_blocks: int = -1
    # 某些场景需要先建立引擎状态，例如先运行 Primer 把公共前缀写入 Prefix Cache。
    # setup_requests 不计入正式请求数、输出 token 数、吞吐和 EngineStats 报告。
    setup_requests: list[RequestSpec] = field(default_factory=list)

    def __post_init__(self):
        if not self.requests:
            raise ValueError("benchmark workload 至少需要一个请求")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len 必须大于 0")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens 必须大于 0")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs 必须大于 0")
        if self.num_kvcache_blocks != -1 and self.num_kvcache_blocks <= 0:
            raise ValueError("num_kvcache_blocks 必须为 -1（自动）或正整数")
        for request in [*self.setup_requests, *self.requests]:
            total_length = len(request.prompt_token_ids) + request.max_tokens
            if total_length > self.max_model_len:
                raise ValueError(
                    f"请求总长度 {total_length} 超过 max_model_len={self.max_model_len}"
                )

    @property
    def prompts(self) -> list[list[int]]:
        """返回可直接传给 LLM.generate() 的 Prompt 列表。"""
        return [request.prompt_token_ids for request in self.requests]

    @property
    def output_lengths(self) -> list[int]:
        """返回与 prompts 顺序一致的最大输出长度。"""
        return [request.max_tokens for request in self.requests]


def _random_prompt(rng: Random, length: int) -> list[int]:
    """生成固定长度的伪随机 token；相同 RNG 状态会得到相同结果。"""
    return [rng.randint(TOKEN_ID_MIN, TOKEN_ID_MAX) for _ in range(length)]


def _build_random_mixed(seed: int) -> WorkloadSpec:
    """保留原 bench.py 的 32 请求随机长度场景，作为兼容的默认 workload。"""
    rng = Random(seed)
    prompts = [
        _random_prompt(rng, rng.randint(100, 256))
        for _ in range(32)
    ]
    output_lengths = [rng.randint(100, 128) for _ in range(32)]
    requests = [
        RequestSpec(prompt, output_length, group="random")
        for prompt, output_length in zip(prompts, output_lengths)
    ]
    return WorkloadSpec(
        name="random_mixed",
        description="兼容原 benchmark 的 32 条随机中等长度请求",
        requests=requests,
        max_model_len=4096,
        max_num_batched_tokens=16384,
        max_num_seqs=512,
    )


def _build_short_prompt_long_decode(seed: int) -> WorkloadSpec:
    """构造 16 条 32-token Prompt、256-token Decode 请求。"""
    rng = Random(seed)
    requests = [
        RequestSpec(_random_prompt(rng, 32), max_tokens=256, group="short")
        for _ in range(16)
    ]
    return WorkloadSpec(
        name="short_prompt_long_decode",
        description="短 Prompt + 长 Decode，主要观察 Decode 吞吐和 ITL",
        requests=requests,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=16,
    )


def _build_long_prompt_short_decode(seed: int) -> WorkloadSpec:
    """构造 8 条 1024-token Prompt、16-token Decode 请求。"""
    rng = Random(seed)
    requests = [
        RequestSpec(_random_prompt(rng, 1024), max_tokens=16, group="long")
        for _ in range(8)
    ]
    return WorkloadSpec(
        name="long_prompt_short_decode",
        description="长 Prompt + 短 Decode，主要观察 Prefill 和 TTFT",
        requests=requests,
        max_model_len=2048,
        max_num_batched_tokens=8192,
        max_num_seqs=8,
    )


def _build_mixed_lengths(seed: int) -> WorkloadSpec:
    """交错构造 8 条短请求和 8 条长请求，固定请求入队顺序。"""
    rng = Random(seed)
    requests = []
    for _ in range(8):
        requests.append(
            RequestSpec(_random_prompt(rng, 32), max_tokens=256, group="short")
        )
        requests.append(
            RequestSpec(_random_prompt(rng, 1024), max_tokens=16, group="long")
        )
    return WorkloadSpec(
        name="mixed_lengths",
        description="长短请求交错入队，观察不同请求组的等待和尾延迟",
        requests=requests,
        max_model_len=2048,
        # 故意小于整批 Prompt token 总数，使当前 Scheduler 分多轮接纳请求。
        max_num_batched_tokens=4096,
        max_num_seqs=16,
    )


def _build_shared_prefix_high_hit(seed: int) -> WorkloadSpec:
    """先用 Primer 建立两个完整 Prefix Block，再让 16 条 Target 复用它们。"""
    rng = Random(seed)
    shared_prefix = _random_prompt(rng, 512)

    # Primer 和 Target 的 32-token 尾部必须彼此不同，避免整条 Prompt 相同后把
    # “共享公共前缀”误测成“完整 Prompt 重放”。循环只处理理论上的随机碰撞。
    unique_tails: list[list[int]] = []
    seen_tails: set[tuple[int, ...]] = set()
    while len(unique_tails) < 17:
        tail = _random_prompt(rng, 32)
        tail_key = tuple(tail)
        if tail_key not in seen_tails:
            seen_tails.add(tail_key)
            unique_tails.append(tail)

    primer = RequestSpec(
        shared_prefix + unique_tails[0],
        max_tokens=1,
        group="primer",
    )
    targets = [
        RequestSpec(
            shared_prefix + unique_tails[index],
            max_tokens=32,
            group="target",
        )
        for index in range(1, 17)
    ]
    return WorkloadSpec(
        name="shared_prefix_high_hit",
        description="Primer 预建 512-token 公共前缀，16 条 Target 复用两个 KV Block",
        requests=targets,
        setup_requests=[primer],
        max_model_len=1024,
        # Primer 会用两轮 Chunked Prefill 完成；Target 命中 512-token 前缀后，
        # 每条只需计算 32-token 尾部，16 条恰好在同一个 512-token step 中完成。
        max_num_batched_tokens=512,
        max_num_seqs=16,
    )


def _build_kv_pressure_preemption(seed: int) -> WorkloadSpec:
    """用两个整 Block Prompt 和两个物理 KV Blocks 稳定触发一次 Decode 抢占。"""
    rng = Random(seed)
    requests = [
        RequestSpec(
            _random_prompt(rng, 256),
            max_tokens=16,
            group="kv_pressure",
        )
        for _ in range(2)
    ]
    return WorkloadSpec(
        name="kv_pressure_preemption",
        description="两个 256-token Prompt 共享两个物理 KV Block，Decode 跨块时触发抢占",
        requests=requests,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=2,
        num_kvcache_blocks=2,
    )


def _build_decode_then_long_prefill(seed: int) -> WorkloadSpec:
    """先运行短 Prompt 的 Decode，再在第 9 轮前加入一个长 Prompt。"""
    rng = Random(seed)
    requests = [
        RequestSpec(
            _random_prompt(rng, 32),
            max_tokens=64,
            group="decode",
            arrival_step=0,
        )
        for _ in range(4)
    ]
    requests.append(
        RequestSpec(
            _random_prompt(rng, 1024),
            max_tokens=16,
            group="late_prefill",
            # Step 0 是初始 Prefill，Step 1～8 是八轮 Decode；该请求在
            # Step 9 执行前进入 waiting，模拟在线用户在 Decode 中途到达。
            arrival_step=9,
        )
    )
    return WorkloadSpec(
        name="decode_then_long_prefill",
        description="四条请求 Decode 中途到达一条长 Prompt，观察 Prefill 对 ITL 的干扰",
        requests=requests,
        max_model_len=2048,
        # 1024-token 迟到 Prompt 必须分成两个 512-token Chunked Prefill Step。
        max_num_batched_tokens=512,
        # 保留常用 CUDA Graph bucket，并允许迟到请求加入后五条请求同时 Decode。
        max_num_seqs=8,
        # 初始四条请求各一块；迟到请求的 1024-token Prompt 占四块，并在首个
        # Decode token 写 KV 时跨入第五块，共需九块。容量固定充足，用于排除
        # Preemption/Recompute 对 ITL 的干扰。
        num_kvcache_blocks=9,
    )


WORKLOAD_BUILDERS = {
    "random_mixed": _build_random_mixed,
    "short_prompt_long_decode": _build_short_prompt_long_decode,
    "long_prompt_short_decode": _build_long_prompt_short_decode,
    "mixed_lengths": _build_mixed_lengths,
    "shared_prefix_high_hit": _build_shared_prefix_high_hit,
    "kv_pressure_preemption": _build_kv_pressure_preemption,
    "decode_then_long_prefill": _build_decode_then_long_prefill,
}
WORKLOAD_NAMES = tuple(WORKLOAD_BUILDERS)

assert WORKLOAD_NAMES == (DEFAULT_WORKLOAD_NAME, *OFFICIAL_WORKLOAD_NAMES)


def build_workload_spec(name: str, seed: int) -> WorkloadSpec:
    """按名字和 seed 构造 workload；未知名字直接报告可用选项。"""
    try:
        builder = WORKLOAD_BUILDERS[name]
    except KeyError as error:
        available = ", ".join(WORKLOAD_NAMES)
        raise ValueError(f"未知 workload：{name}；可用值：{available}") from error
    return builder(seed)
