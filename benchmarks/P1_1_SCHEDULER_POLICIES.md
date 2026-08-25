# P1.1 Scheduler 策略对比

> 本报告记录 `prefill_first`、`decode_first` 和 `time_sliced` 在固定动态到达
> workload 下的正确性与性能证据。性能数字是实验结果，不作为 pytest 的硬断言。

## 实验目的

验证三种 Scheduler 策略可以配置切换，并观察迟到长 Prompt 到达时的取舍：

- `prefill_first`：优先降低新请求 TTFT。
- `decode_first`：优先保护已有请求 ITL。
- `time_sliced=4`：最多连续执行四个 Decode Step，再给 waiting 请求一次 Prefill 机会。

## 实验环境与 Workload

- 日期：2026-08-17
- Git 基线：`92f1b6d`，工作区包含尚未提交的 P1.1 改动
- GPU：NVIDIA GeForce RTX 3060
- 模型：`Qwen3-0.6B`（`bench.py` 默认模型目录）
- stats：开启
- repeat：3，base seed：0
- workload：`decode_then_long_prefill`
- 初始请求：4 条，Prompt 32 token，最多输出 64 token，Step 0 到达
- 迟到请求：1 条，Prompt 1024 token，最多输出 16 token，Step 9 到达
- token budget：512；max sequences：8；KV blocks：9

1024-token 长 Prompt 需要两个 512-token Chunked Prefill Step。KV 容量固定充足，
三种策略均未发生 Preemption 或 Recompute，避免这些事件干扰调度策略对比。

## 运行命令

```bash
python bench.py --enable-stats --workload decode_then_long_prefill \
  --scheduler-policy prefill_first --repeat 3

python bench.py --enable-stats --workload decode_then_long_prefill \
  --scheduler-policy decode_first --repeat 3

python bench.py --enable-stats --workload decode_then_long_prefill \
  --scheduler-policy time_sliced --time-sliced-decode-steps 4 --repeat 3
```

## 结果

吞吐列使用三轮正式运行的中位数；TTFT、ITL、Queue Wait 和 Step 指标来自程序打印的
最后一轮 `seed=2`。因此延迟结果用于同 seed 策略对比，不冒充三轮延迟中位数。

| 策略 | 三轮吞吐 tok/s | 吞吐中位数 | 正常 Decode ITL ms | 被 Prefill 打断 ITL ms | ITL 放大 | 迟到 Prompt TTFT ms | Queue Wait Max ms | ITL P99 ms | Step（P/D） | Peak KV |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `prefill_first` | 346.65 / 549.36 / 541.95 | 541.95 | 5.88 | 78.26 | 13.31x | 71.19 | 0.17 | 78.26 | 66（3/63） | 9 |
| `decode_first` | 291.81 / 464.90 / 464.40 | 464.40 | 5.99 | 6.82 | 1.14x | 409.84 | 338.49 | 7.24 | 81（3/78） | 5 |
| `time_sliced=4` | 324.07 / 548.15 / 540.03 | 540.03 | 6.12 | 40.48 | 6.61x | 94.76 | 0.11 | 44.04 | 66（3/63） | 9 |

相对 `prefill_first`：

- `decode_first` 将被打断 ITL 从 78.26 ms 降到 6.82 ms，但迟到 Prompt TTFT
  从 71.19 ms 增至 409.84 ms，吞吐中位数下降约 14.3%。它等旧请求全部结束后
  才处理新请求，失去了新旧请求重叠 Decode 的 batching 机会，因此 Decode Step
  从 63 增至 78。
- `time_sliced=4` 将被打断 ITL 降到 40.48 ms，约降低 48.3%；迟到 Prompt TTFT
  增至 94.76 ms，约增加 33.1%；吞吐中位数仅下降约 0.35%。在当前 workload 下，
  它以较小的 TTFT 代价缓解了旧请求的长时间停顿，并保持了 batching 效率。

## Starvation 结论

三种策略均完成 5/5 请求，因此这个有限 workload 中没有发生永久 starvation。

- 严格 `prefill_first` 在 waiting 请求持续到达时，存在 Decode starvation 风险。
- 严格 `decode_first` 在 running 请求持续存在时，存在 Prefill starvation 风险；本次
  最大 Queue Wait 338.49 ms 已体现明显等待，只因旧请求最终结束才没有永久饥饿。
- `time_sliced=4` 为阶段间提供有界机会：waiting 非空时，连续四个 Decode Step 后
  必须尝试 Prefill，避免 Prefill/Decode 之间的永久饥饿。它不解决同一队列内部的
  Chunked Prefill 公平性和队首阻塞，这些属于 P1.2/P1.3。

## 正确性证据

- 参数化 CPU 测试覆盖三种策略，验证最终输出、`FINISHED` 状态、waiting/running
  清空、`block_table` 清空、used KV blocks 清空及全部 `ref_count=0`。
- Scheduler 测试：14 passed。
- 完整 CPU/可用环境测试：92 passed，14 skipped。
- 三组 GPU benchmark 均为 5/5 请求完成，Preemption=0，Recompute=0。

## 结论

P1.1 的配置接口、三种策略语义、资源生命周期和 GPU 取舍对比均已验收。
`time_sliced=4` 是当前动态到达 workload 下的推荐折中基线；该结论只适用于本报告
固定模型、硬件和 workload，后续策略修改需使用相同配置重新测量。
