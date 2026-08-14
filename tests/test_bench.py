import pytest
from types import SimpleNamespace

from bench import (
    MAX_INPUT_LEN,
    MAX_OUTPUT_LEN,
    NUM_SEQS,
    build_workload,
    build_sampling_params,
    format_workload,
    parse_args,
    run_setup_requests,
    run_target_requests,
    run_dynamic_requests,
    run_workload_round,
    summarize_dynamic_arrival,
)
from bench_workloads import WORKLOAD_NAMES, build_workload_spec


# 场景：用户直接运行 python bench.py，不传任何统计或重复参数。
# 验证默认保持旧 benchmark 的含义：关闭指标采集、只跑一轮，并从 seed=0 开始。
def test_benchmark_args_default_to_stats_disabled_and_one_run():
    args = parse_args([])

    assert args.enable_stats is False
    assert args.repeat == 1
    assert args.seed == 0
    assert args.workload == "random_mixed"


# 场景：用户要测量开启指标后的吞吐，并希望同一模式连续采样五轮。
# 验证 --enable-stats、--repeat 和 --seed 都会被正确传给 benchmark 主流程。
def test_benchmark_args_accept_stats_repeat_and_seed():
    args = parse_args([
        "--enable-stats",
        "--repeat",
        "5",
        "--seed",
        "20",
        "--workload",
        "short_prompt_long_decode",
    ])

    assert args.enable_stats is True
    assert args.repeat == 5
    assert args.seed == 20
    assert args.workload == "short_prompt_long_decode"


# 场景：重复次数为 0 时 benchmark 根本没有正式测量结果，负数也没有实际意义。
# 验证参数解析阶段直接拒绝这些输入，避免运行到最后才因空列表或错误轮数失败。
@pytest.mark.parametrize("repeat", ["0", "-1"])
def test_benchmark_args_reject_non_positive_repeat(repeat):
    with pytest.raises(SystemExit):
        parse_args(["--repeat", repeat])


# 场景：stats 关闭和开启需要在两个独立进程中使用完全相同的一组输入做 A/B 对照。
# 验证同一 seed 会生成相同 Prompt 与输出长度，而下一轮 seed 会换一组 Prompt，
# 从而既保证跨进程可复现，也避免单进程多轮测试被上一轮 Prefix Cache 干扰。
def test_build_workload_is_reproducible_and_changes_with_seed():
    prompts_a, output_lengths_a = build_workload(7)
    prompts_b, output_lengths_b = build_workload(7)
    prompts_next, _ = build_workload(8)

    assert prompts_a == prompts_b
    assert output_lengths_a == output_lengths_b
    assert prompts_a != prompts_next
    assert len(prompts_a) == NUM_SEQS
    assert len(output_lengths_a) == NUM_SEQS
    assert all(100 <= len(prompt) <= MAX_INPUT_LEN for prompt in prompts_a)
    assert all(100 <= output_length <= MAX_OUTPUT_LEN for output_length in output_lengths_a)


# 场景：P0.4 第一阶段提供原随机场景和三张具名静态试卷。
# 验证命令行公开的 workload 名称固定且完整，避免构造器已经存在但用户无法选择，
# 或帮助信息暴露一个实际没有实现的名称。
def test_static_workload_names_are_available():
    assert WORKLOAD_NAMES == (
        "random_mixed",
        "short_prompt_long_decode",
        "long_prompt_short_decode",
        "mixed_lengths",
        "shared_prefix_high_hit",
        "kv_pressure_preemption",
        "decode_then_long_prefill",
    )


# 场景：调用方传入拼错或尚未实现的 workload 名称时，不能只看到难懂的字典
# KeyError。验证构造入口给出“未知 workload”以及当前所有可用名称，便于用户
# 直接修正命令，也避免错误地退回默认场景后生成一份没有察觉的错误基线。
def test_unknown_workload_reports_available_names():
    with pytest.raises(ValueError) as error:
        build_workload_spec("unknown", seed=0)

    assert "未知 workload：unknown" in str(error.value)
    assert ", ".join(WORKLOAD_NAMES) in str(error.value)


# 场景：短 Prompt + 长 Decode 用 16 条完全同构的请求把主要运行时间放在 Decode。
# 验证每条 Prompt 固定为 32 token、最多生成 256 token、全部属于 short 组，
# 并且模型长度和单轮 token budget 足够在一次 Prefill 中接纳全部 Prompt。
def test_short_prompt_long_decode_workload_shape():
    workload = build_workload_spec("short_prompt_long_decode", seed=7)

    assert len(workload.requests) == 16
    assert [len(request.prompt_token_ids) for request in workload.requests] == [32] * 16
    assert workload.output_lengths == [256] * 16
    assert [request.group for request in workload.requests] == ["short"] * 16
    assert {request.arrival_step for request in workload.requests} == {0}
    assert workload.max_model_len == 512
    assert workload.max_num_batched_tokens == 512


# 场景：长 Prompt + 短 Decode 用 8 条 1024-token Prompt 强化 Prefill/TTFT 压力。
# 验证输出长度只有 16 token，全部请求属于 long 组，并且 8192-token budget
# 恰好可以在同一个 Prefill step 中容纳 8 条 Prompt，便于不同策略做固定对照。
def test_long_prompt_short_decode_workload_shape():
    workload = build_workload_spec("long_prompt_short_decode", seed=7)

    assert len(workload.requests) == 8
    assert [len(request.prompt_token_ids) for request in workload.requests] == [1024] * 8
    assert workload.output_lengths == [16] * 8
    assert [request.group for request in workload.requests] == ["long"] * 8
    assert workload.max_model_len == 2048
    assert workload.max_num_batched_tokens == 8192


# 场景：长短混合 workload 不是先放完一种请求再放另一种，而是按 short、long
# 交错入队，避免队列顺序本身只偏向某一组。验证 16 条请求的长度、输出长度和
# group 都严格交替；4096-token budget 还会让当前 Scheduler 分多轮接纳 Prompt。
def test_mixed_lengths_workload_interleaves_request_groups():
    workload = build_workload_spec("mixed_lengths", seed=7)

    expected_groups = ["short", "long"] * 8
    expected_prompt_lengths = [32, 1024] * 8
    expected_output_lengths = [256, 16] * 8
    assert [request.group for request in workload.requests] == expected_groups
    assert [len(request.prompt_token_ids) for request in workload.requests] == expected_prompt_lengths
    assert workload.output_lengths == expected_output_lengths
    assert workload.max_num_batched_tokens == 4096


# 场景：固定试卷的长度和分组不能随 seed 漂移，但 token 内容需要随 seed 改变，
# 这样 stats 开/关或不同调度策略能做相同输入的 A/B 对照，多轮重复又不会意外
# 复用上一轮完整 Prompt。验证同名同 seed 完全相同，换 seed 后仅内容发生变化。
@pytest.mark.parametrize(
    "workload_name",
    [
        "short_prompt_long_decode",
        "long_prompt_short_decode",
        "mixed_lengths",
    ],
)
def test_static_workloads_are_reproducible_without_changing_shape(workload_name):
    first = build_workload_spec(workload_name, seed=10)
    repeated = build_workload_spec(workload_name, seed=10)
    next_seed = build_workload_spec(workload_name, seed=11)

    assert first.prompts == repeated.prompts
    assert first.output_lengths == repeated.output_lengths
    assert first.prompts != next_seed.prompts
    assert [len(prompt) for prompt in first.prompts] == [
        len(prompt) for prompt in next_seed.prompts
    ]
    assert [request.group for request in first.requests] == [
        request.group for request in next_seed.requests
    ]


# 场景：benchmark 启动时需要把固定试卷结构打印进日志，后续实验才能确认两次
# 运行是否真的使用相同请求分布。验证摘要包含名称、请求数、长度范围、分组和
# token budget，而不是只打印一个无法审计的 workload 名称。
def test_format_workload_reports_reproducible_shape():
    workload = build_workload_spec("mixed_lengths", seed=7)

    assert format_workload(workload) == (
        "Workload: name=mixed_lengths, requests=16, prompt=32..1024tok, "
        "output=16..256tok, groups=long=8,short=8, token_budget=4096, "
        "max_seqs=16, kv_blocks=auto"
    )


# 场景：共享前缀 workload 用一个 Primer 预建缓存，再用 16 个 Target 测量高命中。
# 验证所有请求都由“完全相同的 512-token 公共前缀 + 各自不同的 32-token 尾部”
# 组成；Primer 只生成 1 token，正式 Target 各生成 32 token，并且引擎容量配置
# 可以让命中前缀后的 16 个尾部在一个 Prefill step 中一起处理。
def test_shared_prefix_workload_has_one_primer_and_sixteen_targets():
    workload = build_workload_spec("shared_prefix_high_hit", seed=7)

    assert len(workload.setup_requests) == 1
    assert len(workload.requests) == 16
    primer = workload.setup_requests[0]
    all_requests = [primer, *workload.requests]
    assert primer.group == "primer"
    assert primer.max_tokens == 1
    assert [request.group for request in workload.requests] == ["target"] * 16
    assert workload.output_lengths == [32] * 16
    assert [len(request.prompt_token_ids) for request in all_requests] == [544] * 17
    assert all(
        request.prompt_token_ids[:512] == primer.prompt_token_ids[:512]
        for request in workload.requests
    )
    assert len({tuple(request.prompt_token_ids[512:]) for request in all_requests}) == 17
    assert workload.max_model_len == 1024
    assert workload.max_num_batched_tokens == 512
    assert workload.max_num_seqs == 16


# 场景：相同命令需要重建完全相同的 Primer/Target 试卷，换 seed 后只允许 token
# 内容变化，不能改变请求数量、长度和共享关系。验证同 seed 的 setup 与正式请求
# 完全相同；下一 seed 的内容不同，但仍保持 1 Primer、16 Target 和 512-token 前缀。
def test_shared_prefix_workload_is_reproducible_without_changing_shape():
    first = build_workload_spec("shared_prefix_high_hit", seed=10)
    repeated = build_workload_spec("shared_prefix_high_hit", seed=10)
    next_seed = build_workload_spec("shared_prefix_high_hit", seed=11)

    assert first == repeated
    assert first.setup_requests != next_seed.setup_requests
    assert first.requests != next_seed.requests
    assert [len(request.prompt_token_ids) for request in first.setup_requests] == [544]
    assert [len(request.prompt_token_ids) for request in next_seed.setup_requests] == [544]
    assert [len(request.prompt_token_ids) for request in first.requests] == [544] * 16
    assert [len(request.prompt_token_ids) for request in next_seed.requests] == [544] * 16
    assert all(
        request.prompt_token_ids[:512]
        == next_seed.setup_requests[0].prompt_token_ids[:512]
        for request in next_seed.requests
    )


class FakeSamplingParams:
    """只保存 benchmark 传入的采样字段，避免 CPU 测试加载真实推理依赖。"""

    def __init__(self, temperature, ignore_eos, max_tokens):
        self.temperature = temperature
        self.ignore_eos = ignore_eos
        self.max_tokens = max_tokens


class RecordingLLM:
    """记录每次 generate 的输入，并模拟第二批 generate 后只保留 Target 指标。"""

    def __init__(self):
        self.calls = []
        self.metrics_request_count = 0

    def generate(self, prompts, sampling_params, use_tqdm):
        self.calls.append((prompts, sampling_params, use_tqdm))
        # 真实 LLMEngine.generate() 会在空闲引擎开始新批次时 reset EngineStats。
        self.metrics_request_count = len(prompts)
        return [
            {"token_ids": [0] * params.max_tokens}
            for params in sampling_params
        ]


class RecordingClock:
    """记录同步和读时钟的先后顺序，验证 Primer 位于正式计时边界之外。"""

    def __init__(self, events):
        self.events = events
        self.values = iter([10.0, 12.0])

    def synchronize(self):
        self.events.append("synchronize")

    def clock(self):
        self.events.append("clock")
        return next(self.values)


# 场景：Prefix Cache 只有在 Primer 完整结束后才能供 Target 查询。用记录调用的
# 假 LLM 验证 benchmark 严格发起两个 batch：第一次只有 1 个 Primer，第二次才是
# 16 个 Target；不能把 17 个请求一次性提交，否则 Target 调度时缓存尚未建立。
def test_shared_prefix_workload_runs_primer_before_targets():
    workload = build_workload_spec("shared_prefix_high_hit", seed=7)
    llm = RecordingLLM()

    run_setup_requests(llm, workload, FakeSamplingParams)
    target_params = build_sampling_params(workload.requests, FakeSamplingParams)
    run_target_requests(llm, workload, target_params)

    assert len(llm.calls) == 2
    setup_prompts, setup_params, setup_tqdm = llm.calls[0]
    target_prompts, recorded_target_params, target_tqdm = llm.calls[1]
    assert setup_prompts == [workload.setup_requests[0].prompt_token_ids]
    assert [params.max_tokens for params in setup_params] == [1]
    assert setup_tqdm is False
    assert target_prompts == workload.prompts
    assert [params.max_tokens for params in recorded_target_params] == [32] * 16
    assert target_tqdm is False


# 场景：Primer 是建立缓存的准备成本，不属于正式高命中 workload 的测量对象。
# 验证正式输出、token 总数和最近一批指标都只包含 16 个 Target：总计 16×32=512
# token，而不是把 Primer 算成第 17 个请求和第 513 个正式输出 token。
def test_shared_prefix_workload_excludes_primer_from_formal_results():
    workload = build_workload_spec("shared_prefix_high_hit", seed=7)
    events = []
    llm = RecordingLLM()
    original_generate = llm.generate

    def record_generate(prompts, sampling_params, use_tqdm):
        events.append(f"generate:{len(prompts)}")
        return original_generate(prompts, sampling_params, use_tqdm)

    llm.generate = record_generate
    timing = RecordingClock(events)
    outputs, target_params, elapsed = run_workload_round(
        llm,
        workload,
        FakeSamplingParams,
        synchronize=timing.synchronize,
        clock=timing.clock,
    )

    assert len(outputs) == 16
    assert sum(len(output["token_ids"]) for output in outputs) == 512
    assert sum(params.max_tokens for params in target_params) == 512
    assert llm.metrics_request_count == 16
    assert elapsed == 2.0
    assert events == [
        "generate:1",
        "synchronize",
        "clock",
        "generate:16",
        "synchronize",
        "clock",
    ]


# 场景：benchmark 配置摘要需要公开 Primer 的存在，但 requests、prompt/output 范围
# 仍然只描述正式 Target。这样日志既能审计“两阶段”设置，也不会误读为 17 个正式
# 请求。验证摘要显示 requests=16、target=16，并额外标记 setup=primer=1。
def test_format_shared_prefix_workload_reports_setup_separately():
    workload = build_workload_spec("shared_prefix_high_hit", seed=7)

    assert format_workload(workload) == (
        "Workload: name=shared_prefix_high_hit, requests=16, prompt=544..544tok, "
        "output=32..32tok, groups=target=16, token_budget=512, max_seqs=16, "
        "kv_blocks=auto, "
        "setup=primer=1"
    )


# 场景：P0.4c 需要在不同显存大小的 GPU 上都稳定触发同一次抢占，不能依赖自动
# 计算恰好得到多少 KV Block。验证 workload 固定为两个互不相同的 256-token Prompt、
# 每条生成 16 token，并明确限制为 2 个物理 Block；两条 Prompt 首轮各占一块，
# 下一轮 Decode 跨入第二逻辑块时必然没有空闲块，从而触发抢占。
def test_kv_pressure_workload_has_two_blocks_for_two_full_block_prompts():
    workload = build_workload_spec("kv_pressure_preemption", seed=7)

    assert len(workload.requests) == 2
    assert [len(prompt) for prompt in workload.prompts] == [256, 256]
    assert workload.prompts[0] != workload.prompts[1]
    assert workload.output_lengths == [16, 16]
    assert [request.group for request in workload.requests] == ["kv_pressure"] * 2
    assert workload.max_model_len == 512
    assert workload.max_num_batched_tokens == 512
    assert workload.max_num_seqs == 2
    assert workload.num_kvcache_blocks == 2


# 场景：KV 紧张试卷也必须支持固定 seed 的重复实验。验证同 seed 会重建完全相同
# 的两条 Prompt，换 seed 后 token 内容改变，但请求数、长度、输出长度和 2-Block
# 容量限制保持不变，方便以后对比抢占策略而不更换工作负载。
def test_kv_pressure_workload_is_reproducible_without_changing_capacity():
    first = build_workload_spec("kv_pressure_preemption", seed=10)
    repeated = build_workload_spec("kv_pressure_preemption", seed=10)
    next_seed = build_workload_spec("kv_pressure_preemption", seed=11)

    assert first == repeated
    assert first.prompts != next_seed.prompts
    assert [len(prompt) for prompt in next_seed.prompts] == [256, 256]
    assert next_seed.output_lengths == [16, 16]
    assert next_seed.num_kvcache_blocks == 2


# 场景：P0.4d 用四条短 Prompt 请求建立稳定 Decode，再在执行 Step 9 前加入一条
# 1024-token 长 Prompt。验证前四条请求的 arrival_step=0，迟到请求属于单独分组、
# arrival_step=9；512-token budget 会把它拆成两轮 Chunked Prefill，9 个 KV Blocks
# 又恰好足够容纳全部请求，避免把调度干扰和 KV 抢占混为一谈。
def test_decode_then_long_prefill_workload_has_delayed_arrival_and_safe_capacity():
    workload = build_workload_spec("decode_then_long_prefill", seed=7)

    assert len(workload.requests) == 5
    assert [len(prompt) for prompt in workload.prompts] == [32] * 4 + [1024]
    assert workload.output_lengths == [64] * 4 + [16]
    assert [request.group for request in workload.requests] == ["decode"] * 4 + [
        "late_prefill"
    ]
    assert [request.arrival_step for request in workload.requests] == [0] * 4 + [9]
    assert workload.max_model_len == 2048
    assert workload.max_num_batched_tokens == 512
    assert workload.max_num_seqs == 8
    assert workload.num_kvcache_blocks == 9


# 场景：动态到达试卷也要像其他 workload 一样支持相同命令复现实验。验证相同 seed
# 会重建完全相同的五条请求；更换 seed 只改变 Prompt token，不得改变 4+1 分组、
# Prompt/输出长度、到达 Step 或容量配置，便于未来公平比较三种 Scheduler 策略。
def test_decode_then_long_prefill_is_reproducible_without_changing_arrivals():
    first = build_workload_spec("decode_then_long_prefill", seed=10)
    repeated = build_workload_spec("decode_then_long_prefill", seed=10)
    next_seed = build_workload_spec("decode_then_long_prefill", seed=11)

    assert first == repeated
    assert first.prompts != next_seed.prompts
    assert [len(prompt) for prompt in next_seed.prompts] == [32] * 4 + [1024]
    assert next_seed.output_lengths == [64] * 4 + [16]
    assert [request.arrival_step for request in next_seed.requests] == [0] * 4 + [9]
    assert next_seed.num_kvcache_blocks == 9


class DynamicRecordingLLM:
    """模拟可逐轮驱动的引擎，记录请求究竟在哪一个 Engine Step 被加入。"""

    def __init__(self, completion_step_by_seq_id):
        self.completion_step_by_seq_id = completion_step_by_seq_id
        self.current_step = 0
        self.next_seq_id = 100
        self.active_seq_ids = set()
        self.events = []
        self.reset_count = 0
        self.tokenizer = SimpleNamespace(decode=lambda token_ids: str(token_ids))

    def reset_metrics(self):
        assert not self.active_seq_ids
        self.reset_count += 1
        self.events.append("reset")

    def add_request(self, prompt, sampling_params):
        seq_id = self.next_seq_id
        self.next_seq_id += 1
        self.active_seq_ids.add(seq_id)
        self.events.append(
            ("add", self.current_step, seq_id, len(prompt), sampling_params.max_tokens)
        )
        return seq_id

    def is_finished(self):
        return not self.active_seq_ids

    def step(self):
        self.events.append(("step", self.current_step, tuple(sorted(self.active_seq_ids))))
        completed = []
        for seq_id in sorted(self.active_seq_ids):
            if self.completion_step_by_seq_id[seq_id] == self.current_step:
                completed.append((seq_id, [seq_id]))
        self.active_seq_ids.difference_update(seq_id for seq_id, _ in completed)
        self.current_step += 1
        return completed, -len(self.active_seq_ids)


# 场景：普通 generate() 会在开头一次性提交全部 Prompt，因此无法模拟在线到达。
# 用记录型假引擎验证动态运行器先在 Step 0 加入四条旧请求，完整执行 Step 0～8，
# 然后才在 Step 9 前加入长 Prompt；五条请求结束前循环不会提前退出，最终输出顺序
# 仍按 RequestSpec 顺序排列，而不是按各请求实际完成顺序排列。
def test_dynamic_runner_adds_long_prompt_only_before_step_nine():
    workload = build_workload_spec("decode_then_long_prefill", seed=7)
    sampling_params = build_sampling_params(workload.requests, FakeSamplingParams)
    llm = DynamicRecordingLLM(
        completion_step_by_seq_id={100: 11, 101: 12, 102: 13, 103: 14, 104: 15}
    )

    result = run_dynamic_requests(llm, workload, sampling_params, clock=lambda: 1.0)

    add_events = [event for event in llm.events if event[0] == "add"]
    step_events = [event for event in llm.events if event[0] == "step"]
    assert llm.reset_count == 1
    assert [(event[1], event[3]) for event in add_events] == [(0, 32)] * 4 + [
        (9, 1024)
    ]
    assert [event[1] for event in step_events[:10]] == list(range(10))
    assert [event[1] for event in step_events] == list(range(16))
    assert 104 not in step_events[8][2]
    assert 104 in step_events[9][2]
    assert result.request_seq_ids == [100, 101, 102, 103, 104]
    assert [output["token_ids"] for output in result.outputs] == [
        [100], [101], [102], [103], [104]
    ]


# 场景：开启 stats 后，benchmark 需要把整体 ITL 中真正跨越迟到 Prefill 的间隔
# 单独提取出来。构造四条旧请求在 1/2/3/8 秒输出 token、迟到请求 TTFT=2 秒；
# 验证普通 Decode ITL P50=1 秒，跨到达间隔=5 秒，并且摘要不会把迟到请求自身
# 的 token 间隔混入旧请求的中断指标。
def test_dynamic_arrival_summary_separates_normal_and_interrupted_itl():
    workload = build_workload_spec("decode_then_long_prefill", seed=7)
    request_seq_ids = [10, 11, 12, 13, 14]
    metrics = {
        seq_id: SimpleNamespace(
            output_token_times=[1.0, 2.0, 3.0, 8.0],
            ttft=1.0,
        )
        for seq_id in request_seq_ids[:4]
    }
    metrics[14] = SimpleNamespace(output_token_times=[7.0, 8.0], ttft=2.0)
    llm = SimpleNamespace(get_request_metrics=lambda: metrics)
    result = SimpleNamespace(
        request_seq_ids=request_seq_ids,
        arrival_step_times={0: 0.0, 9: 4.0},
    )

    summary = summarize_dynamic_arrival(llm, workload, result)

    assert summary is not None
    assert summary.format() == (
        "Dynamic Arrival: step=9, normal_decode_itl_p50=1000.00ms, "
        "interrupted_decode_itl_p50=5000.00ms, "
        "interrupted_decode_itl_max=5000.00ms, "
        "late_prefill_ttft_p50=2000.00ms"
    )


# 场景：用户可以用 --disable-stats 只测动态 workload 的吞吐。此时引擎不会创建
# RequestMetrics，但请求仍应正常完成；验证动态摘要直接返回 None，而不是为了读取
# ITL 访问空字典报错，从而保持统计开关只影响观测能力、不改变执行语义。
def test_dynamic_arrival_summary_is_absent_when_stats_are_disabled():
    workload = build_workload_spec("decode_then_long_prefill", seed=7)
    llm = SimpleNamespace(get_request_metrics=lambda: {})
    result = SimpleNamespace(
        request_seq_ids=[10, 11, 12, 13, 14],
        arrival_step_times={0: 0.0, 9: 4.0},
    )

    assert summarize_dynamic_arrival(llm, workload, result) is None


# 场景：动态 workload 的配置摘要必须把到达分布写进日志，避免只看到五个请求却
# 无法确认它们是一起提交还是分两批提交。验证摘要明确显示 Step 0 的四条 Decode
# 请求、Step 9 的一条长 Prefill，以及用于隔离抢占变量的 9-Block 固定容量。
def test_format_dynamic_workload_reports_arrival_schedule():
    workload = build_workload_spec("decode_then_long_prefill", seed=7)

    assert format_workload(workload) == (
        "Workload: name=decode_then_long_prefill, requests=5, "
        "prompt=32..1024tok, output=16..64tok, "
        "groups=decode=4,late_prefill=1, token_budget=512, "
        "max_seqs=8, kv_blocks=9, arrivals=step0=4,step9=1"
    )
