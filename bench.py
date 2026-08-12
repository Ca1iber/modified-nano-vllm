import argparse
import os
import time
from random import Random
from statistics import median

DEFAULT_MODEL_PATH = "~/huggingface/Qwen3-0.6B/"
NUM_SEQS = 32
MAX_INPUT_LEN = 256
MAX_OUTPUT_LEN = 128


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
    return parser.parse_args(argv)


def build_workload(workload_seed: int) -> tuple[list[list[int]], list[int]]:
    """根据固定 seed 构造可复现的 Prompt 和输出长度。"""
    rng = Random(workload_seed)
    prompt_token_ids = [
        [rng.randint(0, 10000) for _ in range(rng.randint(100, MAX_INPUT_LEN))]
        for _ in range(NUM_SEQS)
    ]
    output_lengths = [rng.randint(100, MAX_OUTPUT_LEN) for _ in range(NUM_SEQS)]
    return prompt_token_ids, output_lengths


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # 把 GPU 相关依赖延迟到参数解析之后，使 --help 和参数单测无需加载 PyTorch。
    import torch

    from nanovllm import LLM, SamplingParams

    model_path = os.path.expanduser(args.model)
    llm = LLM(
        model_path,
        enforce_eager=False,
        max_model_len=4096,
        enable_stats=args.enable_stats,
    )

    # 预热不计入正式结果，避免首次 CUDA Graph 捕获和初始化成本污染吞吐。
    torch.manual_seed(args.seed)
    llm.generate(["Benchmark: "], SamplingParams())

    stats_state = "enabled" if args.enable_stats else "disabled"
    print(
        f"Benchmark: stats={stats_state}, repeats={args.repeat}, "
        f"base_seed={args.seed}"
    )

    throughputs: list[float] = []
    last_seed = args.seed
    for run_index in range(args.repeat):
        # 每轮换 seed，避免上一轮完整 Prompt 被 Prefix Cache 命中；相同命令参数
        # 在 stats 开/关的两个进程中仍会生成完全一致的一组工作负载。
        workload_seed = args.seed + run_index
        prompt_token_ids, output_lengths = build_workload(workload_seed)
        sampling_params = [
            SamplingParams(
                temperature=0.6,
                ignore_eos=True,
                max_tokens=output_length,
            )
            for output_length in output_lengths
        ]
        torch.manual_seed(workload_seed)

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        total_tokens = sum(params.max_tokens for params in sampling_params)
        throughput = total_tokens / elapsed
        throughputs.append(throughput)
        last_seed = workload_seed
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


if __name__ == "__main__":
    main()
