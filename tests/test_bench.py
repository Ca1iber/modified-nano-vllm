import pytest

from bench import (
    MAX_INPUT_LEN,
    MAX_OUTPUT_LEN,
    NUM_SEQS,
    build_workload,
    parse_args,
)


# 场景：用户直接运行 python bench.py，不传任何统计或重复参数。
# 验证默认保持旧 benchmark 的含义：关闭指标采集、只跑一轮，并从 seed=0 开始。
def test_benchmark_args_default_to_stats_disabled_and_one_run():
    args = parse_args([])

    assert args.enable_stats is False
    assert args.repeat == 1
    assert args.seed == 0


# 场景：用户要测量开启指标后的吞吐，并希望同一模式连续采样五轮。
# 验证 --enable-stats、--repeat 和 --seed 都会被正确传给 benchmark 主流程。
def test_benchmark_args_accept_stats_repeat_and_seed():
    args = parse_args(["--enable-stats", "--repeat", "5", "--seed", "20"])

    assert args.enable_stats is True
    assert args.repeat == 5
    assert args.seed == 20


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
