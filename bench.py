import argparse
from collections import Counter
import os
import time
from dataclasses import dataclass
from statistics import median

from bench_workloads import (
    DEFAULT_WORKLOAD_NAME,
    WORKLOAD_NAMES,
    RequestSpec,
    WorkloadSpec,
    build_workload_spec,
)
from nanovllm.config import SCHEDULER_POLICIES

DEFAULT_MODEL_PATH = "~/huggingface/Qwen3-0.6B/"
NUM_SEQS = 32
MAX_INPUT_LEN = 256
MAX_OUTPUT_LEN = 128


@dataclass(slots=True)
class WorkloadRunResult:
    """一轮正式 workload 的输出、请求映射和动态到达事件。"""

    outputs: list[dict]
    request_seq_ids: list[int]
    arrival_step_times: dict[int, float]

    def __iter__(self):
        """兼容原来把 run_target_requests() 结果直接当输出列表遍历的调用方。"""
        return iter(self.outputs)

    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        return self.outputs[index]


@dataclass(frozen=True, slots=True)
class DynamicArrivalSummary:
    """迟到 Prefill 对旧 Decode 请求造成的可机器汇总延迟。"""

    arrival_step: int
    normal_decode_itl_p50_ms: float
    interrupted_decode_itl_p50_ms: float
    interrupted_decode_itl_max_ms: float
    late_prefill_ttft_p50_ms: float

    def format(self) -> str:
        return (
            f"Dynamic Arrival: step={self.arrival_step}, "
            f"normal_decode_itl_p50={self.normal_decode_itl_p50_ms:.2f}ms, "
            f"interrupted_decode_itl_p50={self.interrupted_decode_itl_p50_ms:.2f}ms, "
            f"interrupted_decode_itl_max={self.interrupted_decode_itl_max_ms:.2f}ms, "
            f"late_prefill_ttft_p50={self.late_prefill_ttft_p50_ms:.2f}ms"
        )


def positive_int(value: str) -> int:
    """把命令行参数解析为正整数。"""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return parsed_value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行 nano-vLLM 吞吐 benchmark，并可选择是否采集 Engine Metrics。"
    )
    stats_group = parser.add_mutually_exclusive_group()
    stats_group.add_argument(
        "--enable-stats",
        dest="enable_stats",
        action="store_true",
        help="开启请求、Step、抢占和 KV Cache 指标采集。",
    )
    stats_group.add_argument(
        "--disable-stats",
        dest="enable_stats",
        action="store_false",
        help="关闭指标采集，只测基准吞吐（默认）。",
    )
    parser.set_defaults(enable_stats=False)
    parser.add_argument(
        "--repeat",
        type=positive_int,
        default=1,
        help="在同一个引擎实例中运行多少轮正式测试（默认：1）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="第一轮工作负载的随机种子；后续每轮依次加 1（默认：0）。",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_PATH,
        help=f"模型目录（默认：{DEFAULT_MODEL_PATH}）。",
    )
    parser.add_argument(
        "--workload",
        choices=WORKLOAD_NAMES,
        default=DEFAULT_WORKLOAD_NAME,
        help=(
            "选择可复现工作负载；默认 random_mixed 保持原 benchmark 的请求分布。"
        ),
    )
    parser.add_argument(
        "--scheduler-policy",
        choices=SCHEDULER_POLICIES,
        default="prefill_first",
        help="调度顺序：prefill_first、decode_first 或 time_sliced（默认：prefill_first）。",
    )
    parser.add_argument(
        "--time-sliced-decode-steps",
        type=positive_int,
        default=4,
        help="time_sliced 连续 Decode 的 Step 配额（默认：4）。",
    )
    return parser.parse_args(argv)


def build_workload(workload_seed: int) -> tuple[list[list[int]], list[int]]:
    """兼容旧测试和调用方：构造默认 random_mixed workload。"""
    workload = build_workload_spec(DEFAULT_WORKLOAD_NAME, workload_seed)
    return workload.prompts, workload.output_lengths


def format_workload(workload: WorkloadSpec) -> str:
    """把 workload 的固定结构整理成一行可保存的 benchmark 配置摘要。"""
    prompt_lengths = [len(request.prompt_token_ids) for request in workload.requests]
    output_lengths = workload.output_lengths
    groups = Counter(request.group for request in workload.requests)
    group_text = ",".join(
        f"{group}={count}"
        for group, count in sorted(groups.items())
    )
    summary = (
        f"Workload: name={workload.name}, requests={len(workload.requests)}, "
        f"prompt={min(prompt_lengths)}..{max(prompt_lengths)}tok, "
        f"output={min(output_lengths)}..{max(output_lengths)}tok, "
        f"groups={group_text}, token_budget={workload.max_num_batched_tokens}, "
        f"max_seqs={workload.max_num_seqs}, "
        f"kv_blocks={'auto' if workload.num_kvcache_blocks == -1 else workload.num_kvcache_blocks}"
    )
    if workload.setup_requests:
        setup_groups = Counter(request.group for request in workload.setup_requests)
        setup_text = ",".join(
            f"{group}={count}"
            for group, count in sorted(setup_groups.items())
        )
        summary += f", setup={setup_text}"
    if any(request.arrival_step for request in workload.requests):
        arrivals = Counter(request.arrival_step for request in workload.requests)
        arrival_text = ",".join(
            f"step{arrival_step}={count}"
            for arrival_step, count in sorted(arrivals.items())
        )
        summary += f", arrivals={arrival_text}"
    return summary


def build_sampling_params(
    requests: list[RequestSpec],
    sampling_params_type,
) -> list:
    """为一组 RequestSpec 创建顺序和长度严格对应的采样参数。"""
    return [
        sampling_params_type(
            temperature=0.6,
            ignore_eos=True,
            max_tokens=request.max_tokens,
        )
        for request in requests
    ]


def run_setup_requests(llm, workload: WorkloadSpec, sampling_params_type) -> None:
    """在正式计时前运行 Primer 等准备请求；普通 workload 在这里不执行任何操作。"""
    if not workload.setup_requests:
        return
    llm.generate(
        [request.prompt_token_ids for request in workload.setup_requests],
        build_sampling_params(workload.setup_requests, sampling_params_type),
        use_tqdm=False,
    )


def run_dynamic_requests(
    llm,
    workload: WorkloadSpec,
    sampling_params: list,
    clock=time.perf_counter,
) -> WorkloadRunResult:
    """按 arrival_step 在相邻 Engine Step 之间加入请求并驱动到全部完成。"""
    if len(sampling_params) != len(workload.requests):
        raise ValueError("sampling_params 数量必须与 workload.requests 一致")

    pending_requests = sorted(
        enumerate(zip(workload.requests, sampling_params)),
        key=lambda item: (item[1][0].arrival_step, item[0]),
    )
    pending_index = 0
    engine_step = 0
    outputs_by_seq_id = {}
    request_seq_ids = [None] * len(workload.requests)
    arrival_step_times = {}

    # generate() 会在空闲引擎上自动 reset；手动 add_request/step 也需要相同的
    # 正式批次边界，避免把预热或上一轮的 EngineStats 混入本轮。
    llm.reset_metrics()
    while pending_index < len(pending_requests) or not llm.is_finished():
        if llm.is_finished() and pending_index < len(pending_requests):
            next_arrival_step = pending_requests[pending_index][1][0].arrival_step
            if next_arrival_step > engine_step:
                # 没有活跃请求时不执行空 step；直接推进到下一次预定到达。
                engine_step = next_arrival_step

        while pending_index < len(pending_requests):
            request_index, (request, params) = pending_requests[pending_index]
            if request.arrival_step > engine_step:
                break
            if request.arrival_step not in arrival_step_times:
                arrival_step_times[request.arrival_step] = clock()
            request_seq_ids[request_index] = llm.add_request(
                request.prompt_token_ids,
                params,
            )
            pending_index += 1

        if llm.is_finished():
            continue
        completed, _ = llm.step()
        for seq_id, token_ids in completed:
            outputs_by_seq_id[seq_id] = token_ids
        engine_step += 1

    outputs = [
        {
            "text": llm.tokenizer.decode(outputs_by_seq_id[seq_id]),
            "token_ids": outputs_by_seq_id[seq_id],
        }
        for seq_id in request_seq_ids
    ]
    return WorkloadRunResult(
        outputs=outputs,
        request_seq_ids=request_seq_ids,
        arrival_step_times=arrival_step_times,
    )


def run_target_requests(
    llm,
    workload: WorkloadSpec,
    sampling_params: list,
    clock=time.perf_counter,
) -> WorkloadRunResult:
    """运行正式请求；存在非零 arrival_step 时改用手动 add_request/step。"""
    if any(request.arrival_step for request in workload.requests):
        return run_dynamic_requests(llm, workload, sampling_params, clock=clock)

    outputs = llm.generate(
        workload.prompts,
        sampling_params,
        use_tqdm=False,
    )
    get_request_metrics = getattr(llm, "get_request_metrics", None)
    request_metrics = get_request_metrics() if get_request_metrics is not None else {}
    request_seq_ids = list(request_metrics) if request_metrics else []
    return WorkloadRunResult(
        outputs=outputs,
        request_seq_ids=request_seq_ids,
        arrival_step_times={},
    )


def run_workload_round(
    llm,
    workload: WorkloadSpec,
    sampling_params_type,
    synchronize,
    clock,
):
    """完成一轮 setup + 正式 Target，并只返回 Target 的计时与结果。"""
    sampling_params = build_sampling_params(workload.requests, sampling_params_type)
    run_setup_requests(llm, workload, sampling_params_type)
    synchronize()
    start_time = clock()
    result = run_target_requests(llm, workload, sampling_params, clock=clock)
    synchronize()
    elapsed = clock() - start_time
    return result, sampling_params, elapsed


def summarize_dynamic_arrival(
    llm,
    workload: WorkloadSpec,
    result: WorkloadRunResult,
) -> DynamicArrivalSummary | None:
    """汇总迟到 Prefill 对此前 Decode 请求造成的跨到达 ITL。"""
    delayed_steps = sorted(
        {request.arrival_step for request in workload.requests if request.arrival_step}
    )
    if not delayed_steps:
        return None

    request_metrics = llm.get_request_metrics()
    if not request_metrics:
        return None
    delayed_arrival_step = delayed_steps[0]
    delayed_arrival_time = result.arrival_step_times[delayed_arrival_step]
    interruption_itls = []
    normal_itls = []
    for request, seq_id in zip(workload.requests, result.request_seq_ids):
        if request.arrival_step >= delayed_arrival_step:
            continue
        token_times = request_metrics[seq_id].output_token_times
        for previous, current in zip(token_times, token_times[1:]):
            if not (previous <= delayed_arrival_time < current):
                normal_itls.append(current - previous)
        before = [value for value in token_times if value <= delayed_arrival_time]
        after = [value for value in token_times if value > delayed_arrival_time]
        if before and after:
            interruption_itls.append(after[0] - before[-1])

    late_ttfts = [
        request_metrics[seq_id].ttft
        for request, seq_id in zip(workload.requests, result.request_seq_ids)
        if request.arrival_step == delayed_arrival_step
        and request_metrics[seq_id].ttft is not None
    ]
    interruption_ms = [value * 1000 for value in interruption_itls]
    normal_itl_ms = [value * 1000 for value in normal_itls]
    late_ttft_ms = [value * 1000 for value in late_ttfts]
    if not interruption_ms or not normal_itl_ms or not late_ttft_ms:
        return None
    return DynamicArrivalSummary(
        arrival_step=delayed_arrival_step,
        normal_decode_itl_p50_ms=median(normal_itl_ms),
        interrupted_decode_itl_p50_ms=median(interruption_ms),
        interrupted_decode_itl_max_ms=max(interruption_ms),
        late_prefill_ttft_p50_ms=median(late_ttft_ms),
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    base_workload = build_workload_spec(args.workload, args.seed)

    # 把 GPU 相关依赖延迟到参数解析之后，使 --help 和参数单测无需加载 PyTorch。
    import torch

    from nanovllm import LLM, SamplingParams

    model_path = os.path.expanduser(args.model)
    llm = LLM(
        model_path,
        enforce_eager=False,
        max_model_len=base_workload.max_model_len,
        max_num_batched_tokens=base_workload.max_num_batched_tokens,
        max_num_seqs=base_workload.max_num_seqs,
        num_kvcache_blocks=base_workload.num_kvcache_blocks,
        enable_stats=args.enable_stats,
        scheduler_policy=args.scheduler_policy,
        time_sliced_decode_steps=args.time_sliced_decode_steps,
    )

    # 预热不计入正式结果，避免首次 CUDA Graph 捕获和初始化成本污染吞吐。
    torch.manual_seed(args.seed)
    llm.generate(["Benchmark: "], SamplingParams())

    stats_state = "enabled" if args.enable_stats else "disabled"
    print(
        f"Benchmark: workload={args.workload}, scheduler_policy={args.scheduler_policy}, "
        f"stats={stats_state}, repeats={args.repeat}, "
        f"base_seed={args.seed}"
    )
    print(format_workload(base_workload))

    throughputs: list[float] = []
    last_seed = args.seed
    last_dynamic_summary = None
    for run_index in range(args.repeat):
        # 每轮换 seed，避免上一轮完整 Prompt 被 Prefix Cache 命中；相同命令参数
        # 在 stats 开/关的两个进程中仍会生成完全一致的一组工作负载。
        workload_seed = args.seed + run_index
        workload = build_workload_spec(args.workload, workload_seed)
        torch.manual_seed(workload_seed)

        # 共享前缀场景先让 Primer 完整结束，使它产生的完整 KV Blocks 可以被
        # 随后的 Target 命中。Primer 在同步和正式计时之前执行，不计入吞吐。
        result, sampling_params, elapsed = run_workload_round(
            llm,
            workload,
            SamplingParams,
            synchronize=torch.cuda.synchronize,
            clock=time.perf_counter,
        )

        total_tokens = sum(params.max_tokens for params in sampling_params)
        throughput = total_tokens / elapsed
        throughputs.append(throughput)
        last_seed = workload_seed
        last_dynamic_summary = summarize_dynamic_arrival(llm, workload, result)
        print(
            f"Run {run_index + 1}/{args.repeat}: seed={workload_seed}, "
            f"total={total_tokens}tok, time={elapsed:.2f}s, "
            f"throughput={throughput:.2f}tok/s"
        )

    print(
        f"Throughput Summary: runs={len(throughputs)}, "
        f"median={median(throughputs):.2f}tok/s, "
        f"min={min(throughputs):.2f}tok/s, "
        f"max={max(throughputs):.2f}tok/s"
    )

    metrics_summary = llm.get_metrics_summary()
    if metrics_summary is not None:
        # EngineStats 会在每次 generate() 开头清空，因此这里展示的是最后一轮。
        print(f"Last-run metrics (seed={last_seed})")
        print(metrics_summary.format())
        if last_dynamic_summary is not None:
            print(last_dynamic_summary.format())


if __name__ == "__main__":
    main()
