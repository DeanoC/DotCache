# Qwen3.5 MPS Paged Attention 4096 Real-Doc Family Compare

## Coverage

- prompt-file cases: `3`
- replay snapshots: `108`
- full-attention layers: `3, 7, 11, 15, 19, 23`
- kv heads: `0, 1`
- decode steps: `0, 1, 3`

| Case | Prompt | Prefill ms | Dense decode ms/step | Snapshots |
| --- | --- | --- | --- | --- |
| performance_journal | /Users/deanocalver/.codex/worktrees/9f76/DotCache/docs/performance_journal.md | 3851.970 | 97.642 | 36 |
| readme | /Users/deanocalver/.codex/worktrees/9f76/DotCache/README.md | 4217.758 | 122.039 | 36 |
| world_interface_agent_loop_paper | /Users/deanocalver/Documents/world_interface_agent_loop_paper.md | 3957.372 | 101.122 | 36 |

## Normalized Comparison

| View | Scope | Avg ms | Unit | Vs dense decode amortized |
| --- | --- | --- | --- | --- |
| Original Dense Prefill | One 4096-token prompt, full dense model | 4009.033 | ms/prompt | - |
| Original Dense Decode | One generated token, full dense model | 106.934 | ms/step | - |
| Replay Corpus Extraction | One prompt capture: dense prefill + 4 dense decode steps, exporting 36 replay snapshots | 4436.770 | ms/prompt-capture | - |
| Dense Decode Amortized | Decode-side cost per exported layer/head replay snapshot | 8.911 | ms/exported-snapshot | - |
| Replay Extraction Amortized | Full capture cost amortized over exported replay snapshots | 123.244 | ms/exported-snapshot | - |
| Paged Replay Winner | Experimental Backend / Approx Budget | 43.663 | ms/replay-snapshot | 4.900x |
| Paged Replay Fast Tradeoff | Baseline Backend / Approx Budget | 39.940 | ms/replay-snapshot | 4.482x |

## Recommendation

The best fully passing family is `experimental_approx_8_128_c8` (Experimental Backend / Approx Budget) at `43.663 ms`, processing `242.3` tokens with `100.0%` pass rate.

The fastest near-perfect tradeoff is `baseline_approx_8_64_c2` at `39.940 ms` with `99.1%` pass rate.

| Family | Backend / Controller | Avg step ms | Avg tokens | Pass rate | Max abs err | Max rel err |
| --- | --- | --- | --- | --- | --- | --- |
| experimental_approx_8_128_c8 | Experimental Backend / Approx Budget | 43.663 | 242.3 | 100.0% | 0.000004 | 0.000712 |
| baseline_approx_8_64_c2 | Baseline Backend / Approx Budget | 39.940 | 178.3 | 99.1% | 0.000004 | 0.007777 |

## Matched Family Comparison

| Family | Baseline ms | Experimental ms | Speedup | Baseline pass | Experimental pass | Tokens |
| --- | --- | --- | --- | --- | --- | --- |
| approx\|topk=8\|recent=128 | 45.559 | 43.663 | 1.043x | 100.0% | 100.0% | 242.3 |
| robust\|topk=4\|recent=64 | 44.245 | 43.067 | 1.027x | 97.2% | 97.2% | 226.3 |
| approx\|topk=4\|recent=64 | 40.578 | 39.864 | 1.018x | 95.4% | 95.4% | 178.3 |
| approx\|topk=8\|recent=64 | 39.940 | 40.163 | 0.994x | 99.1% | 99.1% | 178.3 |
| approx\|topk=4\|recent=128 | 44.281 | 44.692 | 0.991x | 97.2% | 97.2% | 242.3 |

## All Families

| Family | Backend / Controller | Avg step ms | Avg tokens | Avg pages | Pass rate |
| --- | --- | --- | --- | --- | --- |
| experimental_approx_8_128_c8 | Experimental Backend / Approx Budget | 43.663 | 242.3 | 6.9 | 100.0% |
| baseline_approx_8_128_c2 | Baseline Backend / Approx Budget | 45.559 | 242.3 | 6.9 | 100.0% |
| baseline_approx_8_64_c2 | Baseline Backend / Approx Budget | 39.940 | 178.3 | 5.9 | 99.1% |
| experimental_approx_8_64_c8 | Experimental Backend / Approx Budget | 40.163 | 178.3 | 5.9 | 99.1% |
| experimental_robust_4_64_c8 | Experimental Backend / Robust Full Pass | 43.067 | 226.3 | 8.1 | 97.2% |
| baseline_robust_4_64_c2 | Baseline Backend / Robust Full Pass | 44.245 | 226.3 | 8.1 | 97.2% |
| baseline_approx_4_128_c2 | Baseline Backend / Approx Budget | 44.281 | 242.3 | 6.9 | 97.2% |
| experimental_approx_4_128_c8 | Experimental Backend / Approx Budget | 44.692 | 242.3 | 6.9 | 97.2% |
| experimental_approx_4_64_c8 | Experimental Backend / Approx Budget | 39.864 | 178.3 | 5.9 | 95.4% |
| baseline_approx_4_64_c2 | Baseline Backend / Approx Budget | 40.578 | 178.3 | 5.9 | 95.4% |

