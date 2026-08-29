import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = Path(__file__).with_name("gpu_generation_worker.py")
RESULT_PREFIX = "NANOVLLM_GPU_RESULT="
EXPECTED_OUTPUT_LENGTHS = [8, 6, 4]


def run_generation_worker(
    model_path: Path,
    mode: str,
    workload: str | None = None,
) -> dict:
    """在干净子进程中加载模型，返回 worker 输出的 JSON 结果。"""
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPO_ROOT)
    if old_pythonpath:
        environment["PYTHONPATH"] += os.pathsep + old_pythonpath

    command = [
        sys.executable,
        str(WORKER_PATH),
        "--model",
        str(model_path),
        "--mode",
        mode,
    ]
    if workload is not None:
        command.extend(["--workload", workload])

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if completed.returncode != 0:
        pytest.fail(
            f"{mode} 子进程运行失败（exit={completed.returncode}）\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )

    for line in reversed(completed.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return json.loads(line.removeprefix(RESULT_PREFIX))
    pytest.fail(
        f"{mode} 子进程没有输出 {RESULT_PREFIX} 结果标记\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )


def require_gpu_test_model() -> Path:
    """检查显式开关、CUDA 和模型目录；不满足时给出可见的 skip 原因。"""
    if os.environ.get("NANOVLLM_RUN_GPU_TESTS") != "1":
        pytest.skip("设置 NANOVLLM_RUN_GPU_TESTS=1 后才运行 GPU 正确性测试")
    if not torch.cuda.is_available():
        pytest.skip("当前环境没有可用 CUDA GPU")

    model_path = Path(
        os.environ.get(
            "NANOVLLM_TEST_MODEL",
            "~/huggingface/Qwen3-0.6B/",
        )
    ).expanduser()
    if not model_path.is_dir():
        pytest.skip(f"没有找到 GPU 测试模型目录：{model_path}")
    return model_path


# 场景：相同的三个 Prompt 分别在纯 Eager 和 CUDA Graph 模式下生成 8、6、4 个 token。
# Prefill 在两种模式中本来都走 Eager；从第二个输出 token 开始进入 Decode 后，
# CUDA Graph 一侧会随着请求依次完成经历 batch 3→2→1，并使用 Graph bucket 4/2/1。
# 测试临时使用确定性 argmax，验证两条执行路径的每个最终 token ID 完全一致；
# 这可以发现 Graph replay 前 input、position、slot_mapping、context_lens 或
# block_tables 更新错误，而不会把随机采样差异误判成 CUDA Graph 计算错误。
@pytest.mark.gpu
def test_eager_and_cuda_graph_generate_identical_token_ids():
    model_path = require_gpu_test_model()

    eager_result = run_generation_worker(model_path, mode="eager")
    graph_result = run_generation_worker(model_path, mode="cuda_graph")

    eager_token_ids = eager_result["token_ids"]
    graph_token_ids = graph_result["token_ids"]
    assert [len(tokens) for tokens in eager_token_ids] == EXPECTED_OUTPUT_LENGTHS
    assert [len(tokens) for tokens in graph_token_ids] == EXPECTED_OUTPUT_LENGTHS
    assert eager_token_ids == graph_token_ids


# 场景：目标 Prompt 由“256-token 公共前缀 + 4-token 新尾部”组成。Miss 子进程在
# 干净引擎中完整 Prefill 目标 Prompt；Hit 子进程先用相同前缀、不同尾部的 Primer
# 建立一个完整 Prefix Block，再生成完全相同的目标 Prompt。两边固定使用 Eager 和
# 测试专用 argmax。验证 Miss 的命中量为 0、Hit 恰好命中 1 Block/256 token，
# 并且两条路径最终生成的 4 个 token ID 完全相同，从而证明复用真实 GPU K/V
# 没有改变模型结果，而不是因为实验组没有命中缓存才偶然得到相同输出。
@pytest.mark.gpu
def test_prefix_cache_hit_and_miss_generate_identical_token_ids():
    model_path = require_gpu_test_model()

    miss_result = run_generation_worker(model_path, mode="prefix_miss")
    hit_result = run_generation_worker(model_path, mode="prefix_hit")

    miss_token_ids = miss_result["token_ids"]
    hit_token_ids = hit_result["token_ids"]
    assert [len(tokens) for tokens in miss_token_ids] == [4]
    assert [len(tokens) for tokens in hit_token_ids] == [4]
    assert miss_result["prefix_hit_blocks"] == 0
    assert miss_result["prefix_hit_tokens"] == 0
    assert hit_result["prefix_hit_blocks"] == 1
    assert hit_result["prefix_hit_tokens"] == 256
    assert miss_token_ids == hit_token_ids


# 场景：P0.4b 的真实 benchmark 先单独完成 1 个 Primer，再正式生成 16 个 Target。
# Primer 与 Target 共享两个完整的 256-token Prefix Blocks，但拥有不同的 32-token
# 尾部。验证第二批 EngineStats 已经排除 Primer，只报告 16 个完成请求和 512 个
# 输出 token；同时每个 Target 恰好命中 2 Blocks/512 token，因此总命中必须为
# 32 Blocks/8192 token。该测试证明 CPU 构造出的共享关系真实走进 GPU Prefix Cache。
@pytest.mark.gpu
def test_shared_prefix_workload_hits_expected_gpu_cache_blocks():
    model_path = require_gpu_test_model()

    result = run_generation_worker(model_path, mode="shared_prefix_workload")

    assert result["request_count"] == 16
    assert result["completed_request_count"] == 16
    assert result["output_lengths"] == [32] * 16
    assert result["total_output_tokens"] == 512
    assert result["prefix_hit_blocks"] == 32
    assert result["prefix_hit_tokens"] == 8192


# 场景：相同的两条 256-token Prompt 分别在 4-Block 宽松容量和 2-Block 紧张容量
# 下进行确定性生成。宽松组允许两条请求各自跨入第二逻辑块；紧张组首轮 Prefill
# 后两个 Block 已满，Decode 必须抢占队尾请求，待另一请求完成后恢复并重算。
# 验证紧张组稳定发生一次抢占和 256-token 重算，但两组最终 token ID 完全相同；
# 两组结束时 used=0、全部物理 Block 回到 free，证明抢占没有改变结果或泄漏资源。
@pytest.mark.gpu
def test_kv_pressure_preemption_preserves_output_and_releases_all_blocks():
    model_path = require_gpu_test_model()

    relaxed = run_generation_worker(model_path, mode="kv_pressure_relaxed")
    tight = run_generation_worker(model_path, mode="kv_pressure_tight")

    assert relaxed["request_count"] == relaxed["completed_request_count"] == 2
    assert tight["request_count"] == tight["completed_request_count"] == 2
    assert [len(tokens) for tokens in relaxed["token_ids"]] == [16, 16]
    assert [len(tokens) for tokens in tight["token_ids"]] == [16, 16]
    assert relaxed["token_ids"] == tight["token_ids"]

    assert relaxed["preemptions"] == 0
    assert relaxed["recomputed_tokens"] == 0
    assert relaxed["peak_used_kv_blocks"] == 4
    assert relaxed["used_kv_blocks_after_finish"] == 0
    assert relaxed["free_kv_blocks_after_finish"] == 4

    assert tight["preemptions"] == 1
    assert tight["preempted_cached_tokens"] == 256
    assert tight["released_block_references"] == 1
    assert tight["freed_physical_blocks"] == 1
    assert tight["recompute_steps"] == 1
    assert tight["recomputed_tokens"] == 256
    assert tight["peak_used_kv_blocks"] == 2
    assert tight["used_kv_blocks_after_finish"] == 0
    assert tight["free_kv_blocks_after_finish"] == 2


# 场景：四条 32-token Prompt 在 Step 0 完成 Prefill，并在 Step 1～8 连续 Decode；
# 执行 Step 9 前，一条 1024-token Prompt 才真正加入同一个引擎。当前 prefill_first
# 会暂停旧请求，用两个 512-token Chunked Prefill Step 处理迟到 Prompt，再让五条
# 请求共同 Decode。验证真实 GPU Step 顺序、到达前旧请求各已有 9 个输出 token、
# 5 条请求全部得到规定长度，并且没有 Preemption/Recompute 或 KV Block 泄漏。
@pytest.mark.gpu
def test_dynamic_long_prefill_arrives_during_decode_and_releases_all_blocks():
    model_path = require_gpu_test_model()

    result = run_generation_worker(model_path, mode="dynamic_arrival_workload")

    assert result["request_count"] == result["completed_request_count"] == 5
    assert result["output_lengths"] == [64] * 4 + [16]
    assert result["old_tokens_before_arrival"] == [9] * 4

    assert result["step_num_prefill_tokens"][:12] == (
        [128] + [0] * 8 + [512, 512, 0]
    )
    assert result["step_num_decode_tokens"][:12] == (
        [0] + [4] * 8 + [0, 0, 5]
    )
    assert result["step_num_seqs"][:12] == [4] + [4] * 8 + [1, 1, 5]
    assert result["step_num_tokens"][:12] == [128] + [4] * 8 + [512, 512, 5]

    assert result["preemptions"] == 0
    assert result["recomputed_tokens"] == 0
    assert result["peak_used_kv_blocks"] == 9
    assert result["used_kv_blocks_after_finish"] == 0
    assert result["free_kv_blocks_after_finish"] == 9
