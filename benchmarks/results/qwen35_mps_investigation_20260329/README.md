# Qwen3.5 MPS Investigation

Mac mini first-pass serving investigation for `Qwen/Qwen3.5-0.8B` on `torch_mps`.

## Environment

- Device: Apple MPS
- Backend: `torch_mps`
- Dtype: `float16`
- Tokens per page: `16`
- Decode steps: `2`
- Important caveat: the native flash-linear-attention fast path is not installed on this machine, so these runs used the torch fallback path.

## Raw Files

- [dotcache_exact_m0_mandatory.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_investigation_20260329/dotcache_exact_m0_mandatory.jsonl)
- [dotcache_exact_m0_16384_limit.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_investigation_20260329/dotcache_exact_m0_16384_limit.jsonl)
- [dotcache_local_second_pass_mandatory.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_investigation_20260329/dotcache_local_second_pass_mandatory.jsonl)
- [dense_4096_control.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_mps_investigation_20260329/dense_4096_control.jsonl)

## Results

| Lane | Context | Status | Decode ms/step | Score ms total | Mix ms total | Decode path | Key pages | `K:M2` pages | `V:M1` pages |
|---|---:|---|---:|---:|---:|---|---:|---:|---:|
| Dense | `4096` | ok | `435.23` | n/a | n/a | n/a | n/a | n/a | n/a |
| DotCache exact `M0` | `4096` | ok | `687.45` | `542.32` | `545.13` | `12 grouped / 0 fallback` | `3072` | `0` | `0` |
| DotCache exact `M0` | `8192` | ok | `1320.23` | `1078.25` | `1002.35` | `12 grouped / 0 fallback` | `6144` | `0` | `0` |
| DotCache exact `M0` | `16384` | error | OOM | n/a | n/a | n/a | n/a | n/a | n/a |
| DotCache local second pass | `4096` | ok | `1142.86` | `994.05` | `971.03` | `10 grouped / 2 fallback` | `3072` | `1532` | `4` |
| DotCache local second pass | `8192` | ok | `1354.79` | `1274.55` | `1134.55` | `10 grouped / 2 fallback` | `6144` | `3067` | `4` |
| DotCache local second pass | `16384` | error | OOM | n/a | n/a | n/a | n/a | n/a | n/a |

## Concrete Answers

1. Does MPS still show near-linear long-context growth on the serving path?

Yes for the exact `M0` lane up to the machine limit we reached. Decode time rose from `687.45 ms/step` at `4096` to `1320.23 ms/step` at `8192`, a `1.92x` increase while total static pages doubled from `6144` to `12288`.

We could not validate `16384` or `32768` on this Mac mini because exact `M0` already OOMed at `16384`, so `32768` was not attempted.

2. Does `score + mix` still dominate the backend trace?

Yes. On the successful DotCache runs, `score + mix` remained the overwhelming majority of backend decode time:

- exact `M0` `4096`: `98.68%`
- exact `M0` `8192`: `99.28%`
- second pass `4096`: `97.39%`
- second pass `8192`: `98.35%`

`softmax`, `chunk_assembly`, and `unpack` stayed small by comparison.

3. Does the active mode/profile materially change the bottleneck shape, or mostly just change total page count?

On this Mac, the local second-pass profile changed page mix much more than it changed total page count, and it did not buy a throughput win.

- Total key pages stayed the same as exact `M0` at each context.
- About half the key pages moved from `K:M0` to `K:M2`.
- Values stayed overwhelmingly `V:M0`, with only `4` `V:M1` pages.
- The more important runtime change was decode-path instability: layer `23` fell back to the per-KV path on both successful second-pass runs, while exact `M0` stayed fully grouped.

That path break appears to matter more than the compressed key mix on this machine:

- second pass was much slower than exact `M0` at `4096`
- second pass was still slightly slower than exact `M0` at `8192`
- both lanes OOMed at `16384`

## Extra Notes

- Exact `M0` remained slower than dense at `4096` on this Mac: `687.45 ms/step` vs `435.23 ms/step`.
- In the exact `M0` lane, layer `3` was the heaviest attention-subset layer at both successful contexts, contributing about `21%` of total DotCache decode runtime.
- The second-pass profile made layer `23` disproportionately expensive, consistent with the grouped-to-fallback decode-path split.

## Recommendation

- Treat exact `M0` as the cleaner reference lane for any next MPS investigation.
- If we pursue a model-path shortlist prototype next, start from the exact `M0` lane, not the local second-pass profile.
- Before any algorithmic gating work, it is worth checking why layer `23` stops batching under the second-pass key mix on MPS, because that alone is enough to erase the expected benefit of the profile on this machine.
