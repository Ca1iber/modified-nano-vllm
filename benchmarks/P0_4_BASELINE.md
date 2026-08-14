# P0.4 可复现 Workload GPU 基线

> 本文件由 `bench_suite.py` 生成。性能数字是实验基线，不作为 pytest 的硬断言。
> 完整逐轮结构化指标保存在同名 `.json` 文件中。

## 实验环境

- 日期：2026-08-14
- Git commit：`37417a8`
- 工作区状态：dirty（包含未提交改动）
- GPU：NVIDIA GeForce RTX 3060
- 模型：`/home/czw_ubuntu_wsl/huggingface/Qwen3-0.6B`
- Python：3.10.20
- PyTorch：2.6.0+cu124
- CUDA：12.4
- 执行设置：CUDA Graph，stats 开启，warmup=1，repeat=3，base_seed=0
- 基线状态：valid（正式轮次已通过输出完整性与 Prefix Cache 隔离检查）

## 六场景中位数

| Workload | 吞吐 tok/s | TTFT P50/P99 ms | ITL P50/P99/Max ms | E2E P50/P99 ms | Prefill/Decode Step P50 ms | 抢占 | 重算 token | Prefix hit token | Peak KV Block |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `short_prompt_long_decode` | 1966.55 | 35.85/35.86 | 7.90/10.43/11.76 | 2081.42/2081.43 | 35.84/7.86 | 0 | 0 | 0 | 32 |
| `long_prompt_short_decode` | 164.11 | 620.69/620.82 | 10.89/13.45/13.50 | 779.46/779.46 | 620.82/10.85 | 0 | 0 | 0 | 40 |
| `mixed_lengths` | 953.62 | 343.93/459.23 | 6.92/12.08/300.96 | 1450.82/2280.72 | 170.37/6.83 | 0 | 0 | 0 | 48 |
| `shared_prefix_high_hit` | 1311.69 | 41.11/41.12 | 11.32/13.12/13.12 | 389.58/389.59 | 41.10/11.27 | 0 | 0 | 8192 | 18 |
| `kv_pressure_preemption` | 133.66 | 37.53/37.53 | 5.51/88.64/122.07 | 182.31/237.98 | 36.84/5.46 | 1 | 256 | 0 | 2 |
| `decode_then_long_prefill` | 520.66 | 36.59/70.89 | 6.62/79.34/79.34 | 522.03/522.03 | 36.17/6.56 | 0 | 0 | 0 | 9 |

## 动态到达专项

- 正常 Decode ITL P50：6.45 ms
- 跨越迟到 Prefill 的 ITL P50/Max：79.34/79.34 ms
- 迟到长 Prompt TTFT P50：72.32 ms

## 每轮原始数据

### `short_prompt_long_decode`

Workload: name=short_prompt_long_decode, requests=16, prompt=32..32tok, output=256..256tok, groups=short=16, token_budget=512, max_seqs=16, kv_blocks=auto

| Seed | 耗时 s | 吞吐 tok/s | TTFT P50/P99 ms | ITL P50/P99/Max ms | E2E P50/P99 ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.043 | 2005.38 | 37.03/37.03 | 7.81/10.03/10.63 | 2041.03/2041.03 |
| 1 | 2.273 | 1802.32 | 32.60/32.60 | 8.82/13.65/15.18 | 2271.08/2271.09 |
| 2 | 2.083 | 1966.55 | 35.85/35.86 | 7.90/10.43/11.76 | 2081.42/2081.43 |

### `long_prompt_short_decode`

Workload: name=long_prompt_short_decode, requests=8, prompt=1024..1024tok, output=16..16tok, groups=long=8, token_budget=8192, max_seqs=8, kv_blocks=auto

| Seed | 耗时 s | 吞吐 tok/s | TTFT P50/P99 ms | ITL P50/P99/Max ms | E2E P50/P99 ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.780 | 164.11 | 620.69/620.82 | 9.98/13.45/13.50 | 779.46/779.46 |
| 1 | 0.723 | 177.15 | 556.07/556.18 | 10.89/14.57/14.57 | 722.07/722.08 |
| 2 | 0.815 | 157.13 | 648.35/648.48 | 11.51/12.36/12.36 | 814.12/814.13 |

### `mixed_lengths`

Workload: name=mixed_lengths, requests=16, prompt=32..1024tok, output=16..256tok, groups=long=8,short=8, token_budget=4096, max_seqs=16, kv_blocks=auto

| Seed | 耗时 s | 吞吐 tok/s | TTFT P50/P99 ms | ITL P50/P99/Max ms | E2E P50/P99 ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 2.278 | 955.15 | 340.46/458.63 | 6.92/12.08/300.96 | 1448.58/2277.21 |
| 1 | 2.282 | 953.62 | 343.93/459.23 | 6.91/11.85/300.53 | 1450.82/2280.72 |
| 2 | 2.344 | 928.26 | 352.97/470.75 | 7.11/14.72/308.73 | 1491.31/2343.16 |

### `shared_prefix_high_hit`

Workload: name=shared_prefix_high_hit, requests=16, prompt=544..544tok, output=32..32tok, groups=target=16, token_budget=512, max_seqs=16, kv_blocks=auto, setup=primer=1

| Seed | 耗时 s | 吞吐 tok/s | TTFT P50/P99 ms | ITL P50/P99/Max ms | E2E P50/P99 ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.390 | 1311.37 | 39.97/39.98 | 11.32/13.12/13.12 | 389.58/389.59 |
| 1 | 0.390 | 1311.69 | 41.23/41.24 | 11.31/13.28/13.29 | 389.73/389.73 |
| 2 | 0.385 | 1330.64 | 41.11/41.12 | 11.55/13.04/13.04 | 383.94/383.94 |

### `kv_pressure_preemption`

Workload: name=kv_pressure_preemption, requests=2, prompt=256..256tok, output=16..16tok, groups=kv_pressure=2, token_budget=512, max_seqs=2, kv_blocks=2

| Seed | 耗时 s | 吞吐 tok/s | TTFT P50/P99 ms | ITL P50/P99/Max ms | E2E P50/P99 ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.238 | 134.61 | 37.02/37.02 | 5.36/88.50/121.71 | 179.73/236.27 |
| 1 | 0.239 | 133.66 | 37.53/37.53 | 5.57/88.64/122.07 | 182.31/237.98 |
| 2 | 0.246 | 130.16 | 40.44/40.45 | 5.51/90.18/123.93 | 186.67/244.39 |

### `decode_then_long_prefill`

Workload: name=decode_then_long_prefill, requests=5, prompt=32..1024tok, output=16..64tok, groups=decode=4,late_prefill=1, token_budget=512, max_seqs=8, kv_blocks=9, arrivals=step0=4,step9=1

| Seed | 耗时 s | 吞吐 tok/s | TTFT P50/P99 ms | ITL P50/P99/Max ms | E2E P50/P99 ms |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.522 | 520.66 | 34.31/72.82 | 6.60/81.93/81.93 | 522.03/522.03 |
| 1 | 0.530 | 512.80 | 36.59/70.89 | 6.64/79.34/79.34 | 530.08/530.08 |
| 2 | 0.514 | 529.22 | 36.78/66.28 | 6.62/74.76/74.77 | 513.60/513.60 |
