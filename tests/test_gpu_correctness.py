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


def run_generation_worker(model_path: Path, mode: str) -> dict:
    """在干净子进程中加载模型，返回 worker 输出的 JSON 结果。"""
    environment = os.environ.copy()
    old_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPO_ROOT)
    if old_pythonpath:
        environment["PYTHONPATH"] += os.pathsep + old_pythonpath

    completed = subprocess.run(
        [
            sys.executable,
            str(WORKER_PATH),
            "--model",
            str(model_path),
            "--mode",
            mode,
        ],
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
