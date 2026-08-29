from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from time import perf_counter


@dataclass(frozen=True, slots=True)
class PercentileSummary:
    """一组数值样本的数量、范围和常用延迟百分位。"""

    count: int
    minimum: float | None
    p50: float | None
    p95: float | None
    p99: float | None
    maximum: float | None


def _linear_percentile(sorted_values: list[float], fraction: float) -> float:
    """在相邻样本间线性插值；调用方保证输入非空、有序且 fraction 位于 0～1。"""
    position = (len(sorted_values) - 1) * fraction
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * weight


def summarize_percentiles(values: Iterable[float]) -> PercentileSummary:
    """对任意数值样本计算固定口径的 P50/P95/P99；空样本不伪造为零。"""
    sorted_values = sorted(values)
    if not sorted_values:
        return PercentileSummary(
            count=0,
            minimum=None,
            p50=None,
            p95=None,
            p99=None,
            maximum=None,
        )
    return PercentileSummary(
        count=len(sorted_values),
        minimum=sorted_values[0],
        p50=_linear_percentile(sorted_values, 0.50),
        p95=_linear_percentile(sorted_values, 0.95),
        p99=_linear_percentile(sorted_values, 0.99),
        maximum=sorted_values[-1],
    )


# 一个 RequestMetrics 对应一个 Sequence，通过 seq_id 与请求关联。
# 它保存“这一个请求”的时间线，以及抢占、重算和 Prefix Cache 累计数据；
# TTFT、ITL 和 E2E 等派生指标仍由原始事件动态计算。
@dataclass(slots=True)
class RequestMetrics:
    """一个请求从进入 waiting 到结束为止的时间线。"""

    # 原始事件只记录事实，不提前保存 TTFT、ITL 等可由它们推导出的结果。
    queue_arrival_time: float
    scheduled_times: list[float] = field(default_factory=list)
    output_token_times: list[float] = field(default_factory=list)
    request_finish_time: float | None = None
    # 以下计数均为请求生命周期内的累计值，用于比较不同抢占与缓存策略。
    preemption_count: int = 0
    preempted_cached_tokens: int = 0
    released_block_references: int = 0
    freed_physical_blocks: int = 0
    recompute_steps: int = 0
    recomputed_tokens: int = 0
    prefix_hit_blocks: int = 0
    prefix_hit_tokens: int = 0

    @property
    def first_scheduled_time(self) -> float | None:
        """请求第一次真正进入 scheduled_seqs 的时间。"""
        return self.scheduled_times[0] if self.scheduled_times else None

    @property
    def first_token_time(self) -> float | None:
        """第一个被 Sequence 接受的真实输出 token 的时间。"""
        return self.output_token_times[0] if self.output_token_times else None

    @property
    def last_token_time(self) -> float | None:
        """最近一个被 Sequence 接受的真实输出 token 的时间。"""
        return self.output_token_times[-1] if self.output_token_times else None

    @property
    def queue_wait_time(self) -> float | None:
        """从进入 waiting 到第一次被调度之间的等待时间。"""
        if self.first_scheduled_time is None:
            return None
        return self.first_scheduled_time - self.queue_arrival_time

    @property
    def ttft(self) -> float | None:
        """Time To First Token：从进入 waiting 到首个真实输出 token 的时间。"""
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.queue_arrival_time

    @property
    def itls(self) -> list[float]:
        """Inter-Token Latencies：相邻两个真实输出 token 的时间间隔。"""
        return [
            current - previous
            for previous, current in zip(
                self.output_token_times,
                self.output_token_times[1:],
            )
        ]

    @property
    def e2e_latency(self) -> float | None:
        """从进入 waiting 到请求被标记为 FINISHED 的端到端时间。"""
        if self.request_finish_time is None:
            return None
        return self.request_finish_time - self.queue_arrival_time


# 一个 StepMetrics 对应一次 LLMEngine.step()，而不是对应某个 Sequence。
# 它记录这一整轮的 Prefill/Decode token 组成、driver 侧总耗时、缓存事件，以及
# postprocess 完成后的 used/free 物理 KV Block 快照。
@dataclass(slots=True)
class StepMetrics:
    """一次完整 Engine Step 的工作量组成和耗时。"""

    start_time: float
    end_time: float
    num_seqs: int
    num_prefill_tokens: int
    num_decode_tokens: int
    num_preemptions: int = 0
    num_recomputed_tokens: int = 0
    num_prefix_hit_tokens: int = 0
    used_kv_blocks: int = 0
    free_kv_blocks: int = 0

    @property
    def duration(self) -> float:
        """本轮从 schedule() 前到 postprocess() 后经过的时间。"""
        return self.end_time - self.start_time

    @property
    def num_tokens(self) -> int:
        """本轮实际处理的 Prefill 与 Decode token 总数。"""
        return self.num_prefill_tokens + self.num_decode_tokens

    @property
    def is_prefill_only(self) -> bool:
        """本轮只包含 Prefill token。"""
        return self.num_prefill_tokens > 0 and self.num_decode_tokens == 0

    @property
    def is_decode_only(self) -> bool:
        """本轮只包含 Decode token。"""
        return self.num_decode_tokens > 0 and self.num_prefill_tokens == 0

    @property
    def is_mixed(self) -> bool:
        """本轮同时包含 Prefill 与 Decode token。"""
        return self.num_prefill_tokens > 0 and self.num_decode_tokens > 0

    @property
    def tokens_per_second(self) -> float | None:
        """本轮处理 token 数除以本轮耗时；零耗时时不计算吞吐。"""
        if self.duration == 0:
            return None
        return self.num_tokens / self.duration


@dataclass(frozen=True, slots=True)
class EngineMetricsSummary:
    """从请求时间线和 Step 记录汇总出的 benchmark 结果。"""

    request_count: int
    completed_request_count: int
    step_count: int
    prefill_step_count: int
    decode_step_count: int
    mixed_step_count: int
    queue_wait: PercentileSummary
    ttft: PercentileSummary
    itl: PercentileSummary
    e2e_latency: PercentileSummary
    prefill_step_duration: PercentileSummary
    decode_step_duration: PercentileSummary
    mixed_step_duration: PercentileSummary
    total_preemptions: int
    total_preempted_cached_tokens: int
    total_released_block_references: int
    total_freed_physical_blocks: int
    total_recompute_steps: int
    total_recomputed_tokens: int
    total_prefix_hit_blocks: int
    total_prefix_hit_tokens: int
    peak_used_kv_blocks: int

    @staticmethod
    def _format_milliseconds(name: str, summary: PercentileSummary) -> str:
        """把内部以秒保存的延迟转换成便于阅读的毫秒文本。"""
        if summary.count == 0:
            return f"{name} (ms): count=0"
        return (
            f"{name} (ms): count={summary.count} "
            f"P50={summary.p50 * 1000:.2f} "
            f"P95={summary.p95 * 1000:.2f} "
            f"P99={summary.p99 * 1000:.2f} "
            f"Min={summary.minimum * 1000:.2f} "
            f"Max={summary.maximum * 1000:.2f}"
        )

    def format(self) -> str:
        """生成可由 bench.py 直接打印的多行汇总报告。"""
        return "\n".join(
            [
                "Engine Metrics Summary",
                f"Requests: {self.request_count} (completed={self.completed_request_count})",
                (
                    f"Steps: {self.step_count} "
                    f"(prefill={self.prefill_step_count}, "
                    f"decode={self.decode_step_count}, mixed={self.mixed_step_count})"
                ),
                self._format_milliseconds("Queue Wait", self.queue_wait),
                self._format_milliseconds("TTFT", self.ttft),
                self._format_milliseconds("ITL", self.itl),
                self._format_milliseconds("E2E", self.e2e_latency),
                self._format_milliseconds("Prefill Step", self.prefill_step_duration),
                self._format_milliseconds("Decode Step", self.decode_step_duration),
                self._format_milliseconds("Mixed Step", self.mixed_step_duration),
                f"Preemptions: {self.total_preemptions}",
                f"Preempted cached tokens: {self.total_preempted_cached_tokens}",
                f"Released block references: {self.total_released_block_references}",
                f"Freed physical blocks: {self.total_freed_physical_blocks}",
                f"Recompute steps: {self.total_recompute_steps}",
                f"Recomputed tokens: {self.total_recomputed_tokens}",
                f"Prefix hit blocks: {self.total_prefix_hit_blocks}",
                f"Prefix hit tokens: {self.total_prefix_hit_tokens}",
                f"Peak used KV blocks: {self.peak_used_kv_blocks}",
            ]
        )


# 整个 LLMEngine 只创建一个 EngineStats，统一保管两类统计数据：
# requests 字典按 seq_id 保存每个请求的 RequestMetrics；steps 列表按执行顺序保存
# 每一轮的 StepMetrics。Scheduler 和 LLMEngine 把事件告诉它，再由它读取时钟并记录。
class EngineStats:
    """集中收集请求级时间线和每轮 Engine Step 指标。"""

    def __init__(self, clock: Callable[[], float] = perf_counter):
        # 生产环境使用 perf_counter；测试可以注入 FakeClock，避免依赖真实等待时间。
        self.clock = clock
        self.requests: dict[int, RequestMetrics] = {}
        self.steps: list[StepMetrics] = []
        # 记录每个请求曾经计算到的最远 KV token 位置。抢占后再次 Prefill 时，
        # 只有落在该边界以内、且没有被 Prefix Cache 命中的 token 才算重计算。
        self._recompute_until_tokens: dict[int, int] = {}
        self.begin_step()

    def reset(self):
        """开始一批新的离线生成前，清空上一批请求和 Step 的统计结果。"""
        self.requests.clear()
        self.steps.clear()
        self._recompute_until_tokens.clear()
        self.begin_step()

    def begin_step(self):
        """清空即将开始的一轮 Step 的事件计数，保留请求生命周期累计值。"""
        self._step_num_preemptions = 0
        self._step_num_recomputed_tokens = 0
        self._step_num_prefix_hit_tokens = 0

    def record_arrival(self, seq_id: int):
        """记录请求成功加入 Scheduler.waiting 的时刻。"""
        if seq_id in self.requests:
            raise ValueError(f"sequence {seq_id} already has request metrics")
        self.requests[seq_id] = RequestMetrics(queue_arrival_time=self.clock())
        self._recompute_until_tokens[seq_id] = 0

    def record_scheduled(self, seq_ids: list[int]):
        """记录本轮 scheduled_seqs；同一个 batch 中的请求共享一个时间点。"""
        scheduled_time = self.clock()
        for seq_id in seq_ids:
            self.requests[seq_id].scheduled_times.append(scheduled_time)

    def record_output_token(self, seq_id: int, finished: bool):
        """记录一个被接受的真实 token；若请求结束，同时固定 finish 时间。"""
        output_time = self.clock()
        metrics = self.requests[seq_id]
        metrics.output_token_times.append(output_time)
        if finished:
            metrics.request_finish_time = output_time

    def record_preemption(
        self,
        seq_id: int,
        cached_tokens: int,
        released_block_references: int,
        freed_physical_blocks: int,
    ):
        """累计一次抢占造成的 KV token、Block 引用和物理 Block 损失。"""
        metrics = self.requests[seq_id]
        metrics.preemption_count += 1
        metrics.preempted_cached_tokens += cached_tokens
        metrics.released_block_references += released_block_references
        metrics.freed_physical_blocks += freed_physical_blocks
        # 多次抢占时保留历史最远计算位置，避免漏算第一次抢占尚未恢复的 token。
        self._recompute_until_tokens[seq_id] = max(
            self._recompute_until_tokens[seq_id],
            cached_tokens,
        )
        self._step_num_preemptions += 1

    def record_prefix_cache_hit(self, seq_id: int, num_blocks: int, block_size: int):
        """累计请求本次分配真正复用的完整 Prefix Cache Block 和 token。"""
        if num_blocks == 0:
            return
        hit_tokens = num_blocks * block_size
        metrics = self.requests[seq_id]
        metrics.prefix_hit_blocks += num_blocks
        metrics.prefix_hit_tokens += hit_tokens
        self._step_num_prefix_hit_tokens += hit_tokens

    def record_recompute(
        self,
        seq_id: int,
        scheduled_start: int,
        num_scheduled_tokens: int,
    ) -> int:
        """记录本轮 Prefill 中与抢占前已计算 KV 区间重叠的 token 数。"""
        scheduled_end = scheduled_start + num_scheduled_tokens
        recompute_until = self._recompute_until_tokens[seq_id]
        recomputed_tokens = max(
            0,
            min(scheduled_end, recompute_until) - scheduled_start,
        )
        if recomputed_tokens:
            metrics = self.requests[seq_id]
            metrics.recompute_steps += 1
            metrics.recomputed_tokens += recomputed_tokens
            self._step_num_recomputed_tokens += recomputed_tokens
        return recomputed_tokens

    def record_step(
        self,
        start_time: float,
        num_seqs: int,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        used_kv_blocks: int = 0,
        free_kv_blocks: int = 0,
    ):
        """在 postprocess 完成后，保存这一轮 Engine Step 的完整记录。"""
        self.steps.append(
            StepMetrics(
                start_time=start_time,
                end_time=self.clock(),
                num_seqs=num_seqs,
                num_prefill_tokens=num_prefill_tokens,
                num_decode_tokens=num_decode_tokens,
                num_preemptions=self._step_num_preemptions,
                num_recomputed_tokens=self._step_num_recomputed_tokens,
                num_prefix_hit_tokens=self._step_num_prefix_hit_tokens,
                used_kv_blocks=used_kv_blocks,
                free_kv_blocks=free_kv_blocks,
            )
        )

    def summarize(self) -> EngineMetricsSummary:
        """将当前批次的请求级和 Step 级原始记录整理成统一 benchmark 摘要。"""
        request_metrics = list(self.requests.values())

        # 一个请求最多贡献一个 Queue Wait、TTFT 和 E2E 样本；尚未发生的事件跳过，
        # 不能用 0 填补，否则会人为拉低百分位延迟。
        queue_wait_samples = []
        ttft_samples = []
        e2e_samples = []
        itl_samples = []
        for metrics in request_metrics:
            if metrics.queue_wait_time is not None:
                queue_wait_samples.append(metrics.queue_wait_time)
            if metrics.ttft is not None:
                ttft_samples.append(metrics.ttft)
            if metrics.e2e_latency is not None:
                e2e_samples.append(metrics.e2e_latency)
            # 每对相邻输出 token 都是一条独立 ITL 样本，不能先按请求求平均。
            itl_samples.extend(metrics.itls)

        prefill_steps = [step for step in self.steps if step.is_prefill_only]
        decode_steps = [step for step in self.steps if step.is_decode_only]
        mixed_steps = [step for step in self.steps if step.is_mixed]

        return EngineMetricsSummary(
            request_count=len(request_metrics),
            completed_request_count=sum(
                metrics.request_finish_time is not None
                for metrics in request_metrics
            ),
            step_count=len(self.steps),
            prefill_step_count=len(prefill_steps),
            decode_step_count=len(decode_steps),
            mixed_step_count=len(mixed_steps),
            queue_wait=summarize_percentiles(queue_wait_samples),
            ttft=summarize_percentiles(ttft_samples),
            itl=summarize_percentiles(itl_samples),
            e2e_latency=summarize_percentiles(e2e_samples),
            prefill_step_duration=summarize_percentiles(
                step.duration for step in prefill_steps
            ),
            decode_step_duration=summarize_percentiles(
                step.duration for step in decode_steps
            ),
            mixed_step_duration=summarize_percentiles(
                step.duration for step in mixed_steps
            ),
            total_preemptions=sum(
                metrics.preemption_count for metrics in request_metrics
            ),
            total_preempted_cached_tokens=sum(
                metrics.preempted_cached_tokens for metrics in request_metrics
            ),
            total_released_block_references=sum(
                metrics.released_block_references for metrics in request_metrics
            ),
            total_freed_physical_blocks=sum(
                metrics.freed_physical_blocks for metrics in request_metrics
            ),
            total_recompute_steps=sum(
                metrics.recompute_steps for metrics in request_metrics
            ),
            total_recomputed_tokens=sum(
                metrics.recomputed_tokens for metrics in request_metrics
            ),
            total_prefix_hit_blocks=sum(
                metrics.prefix_hit_blocks for metrics in request_metrics
            ),
            total_prefix_hit_tokens=sum(
                metrics.prefix_hit_tokens for metrics in request_metrics
            ),
            peak_used_kv_blocks=max(
                (step.used_kv_blocks for step in self.steps),
                default=0,
            ),
        )
