#!/usr/bin/env bash

set -euo pipefail

# 无论从仓库根目录还是 tests 目录调用，都用脚本自身位置定位测试文件。
TESTS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

print_usage() {
    echo "用法：bash tests/run.sh <test_name|all> [pytest 参数...]"
    echo
    echo "示例："
    echo "  bash tests/run.sh test_block_manager"
    echo "  bash tests/run.sh test_step_metrics -q"
    echo "  bash tests/run.sh all -q"
    echo
    echo "可用测试："
    for test_file in "${TESTS_DIR}"/test_*.py; do
        test_name="$(basename "${test_file}" .py)"
        echo "  ${test_name}"
    done
    echo "  all"
}

TEST_NAME="${1:-}"
if [[ -z "${TEST_NAME}" ]]; then
    print_usage
    exit 2
fi
shift

# 允许 test_scheduler 和 test_scheduler.py 两种写法。
TEST_NAME="${TEST_NAME%.py}"

if [[ "${TEST_NAME}" == "all" ]]; then
    TEST_TARGET="${TESTS_DIR}"
elif [[ "${TEST_NAME}" =~ ^test_[A-Za-z0-9_]+$ ]]; then
    TEST_TARGET="${TESTS_DIR}/${TEST_NAME}.py"
    if [[ ! -f "${TEST_TARGET}" ]]; then
        echo "错误：没有找到测试 ${TEST_NAME}" >&2
        echo >&2
        print_usage >&2
        exit 2
    fi
else
    echo "错误：测试名必须使用 test_xxx 格式，或者传入 all。" >&2
    echo >&2
    print_usage >&2
    exit 2
fi

# 后续参数原样交给 pytest，例如 -q、-k chunked 或 --maxfail=1。
python -m pytest "${TEST_TARGET}" -v "$@"
