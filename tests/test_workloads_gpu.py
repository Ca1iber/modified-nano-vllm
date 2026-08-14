"""P0.4 六张正式 workload 的统一真实 GPU 验收。"""

import pytest

from bench_workloads import OFFICIAL_WORKLOAD_NAMES
from tests.test_gpu_correctness import require_gpu_test_model, run_generation_worker


EXPECTED_STRUCTURE = {
    "short_prompt_long_decode": {
        "requests": 16,
        "prefill_steps": 1,
        "decode_steps": 255,
        "preemptions": 0,
        "recomputed_tokens": 0,
        "prefix_hit_tokens": 0,
    },
    "long_prompt_short_decode": {
        "requests": 8,
        "prefill_steps": 1,
        "decode_steps": 15,
        "preemptions": 0,
        "recomputed_tokens": 0,
        "prefix_hit_tokens": 0,
    },
    "mixed_lengths": {
        "requests": 16,
        "prefill_steps": 3,
        "decode_steps": 255,
        "preemptions": 0,
        "recomputed_tokens": 0,
        "prefix_hit_tokens": 0,
    },
    "shared_prefix_high_hit": {
        "requests": 16,
        "prefill_steps": 1,
        "decode_steps": 31,
        "preemptions": 0,
        "recomputed_tokens": 0,
        "prefix_hit_tokens": 8192,
    },
    "kv_pressure_preemption": {
        "requests": 2,
        "prefill_steps": 2,
        "decode_steps": 29,
        "preemptions": 1,
        "recomputed_tokens": 256,
        "prefix_hit_tokens": 0,
    },
    "decode_then_long_prefill": {
        "requests": 5,
        "prefill_steps": 3,
        "decode_steps": 63,
        "preemptions": 0,
        "recomputed_tokens": 0,
        "prefix_hit_tokens": 0,
    },
}


# 场景：P0.4 的六张试卷以后会反复用于 Scheduler、KV Cache 和 Runtime 改动。
# 每个参数都启动干净 GPU 子进程，验证正式请求全部完成、输出长度严格符合试卷、
# Prefill/Decode Step 结构与当前基线一致、缓存事件符合场景设计，并且结束后没有
# 物理 KV Block 仍处于 used。耗时与吞吐容易受 GPU 波动影响，不作为 pytest 断言。
@pytest.mark.gpu
@pytest.mark.parametrize("workload_name", OFFICIAL_WORKLOAD_NAMES)
def test_official_workload_completes_with_expected_gpu_state(workload_name):
    model_path = require_gpu_test_model()

    result = run_generation_worker(
        model_path,
        mode="official_workload",
        workload=workload_name,
    )
    expected = EXPECTED_STRUCTURE[workload_name]

    assert result["workload"] == workload_name
    assert result["request_count"] == expected["requests"]
    assert result["completed_request_count"] == expected["requests"]
    assert result["output_lengths"] == result["expected_output_lengths"]
    assert result["prefill_steps"] == expected["prefill_steps"]
    assert result["decode_steps"] == expected["decode_steps"]
    assert result["preemptions"] == expected["preemptions"]
    assert result["recomputed_tokens"] == expected["recomputed_tokens"]
    assert result["prefix_hit_tokens"] == expected["prefix_hit_tokens"]
    assert result["used_kv_blocks_after_finish"] == 0
    assert result["free_kv_blocks_after_finish"] >= result["peak_used_kv_blocks"]

    if workload_name == "decode_then_long_prefill":
        assert result["old_tokens_before_arrival"] == [9] * 4
        assert result["step_is_prefill"][:12] == [
            True,
            False, False, False, False, False, False, False, False,
            True, True,
            False,
        ]
        assert result["step_num_seqs"][:12] == [4] + [4] * 8 + [1, 1, 5]
        assert result["step_num_tokens"][:12] == [128] + [4] * 8 + [512, 512, 5]
