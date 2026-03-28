# Local Layer Profiles

These are first-pass handwritten layer sensitivity profiles derived from local Apple MPS inspection runs. They are meant to be useful hints while the CUDA box is busy, not final production policies.

The two supporting tools are:

- [bench_layer_sensitivity.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_layer_sensitivity.py)
- [inspect_policy_prefill.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/inspect_policy_prefill.py)

The current profile artifacts live in:

- [tinyllama_local_first_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/tinyllama_local_first_pass.yaml)
- [smollm2_360m_local_first_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/smollm2_360m_local_first_pass.yaml)

## TinyLlama

Local prefill inspection at exact `577` prompt tokens with `balanced` K/V policy tiers showed:

- keys stayed overwhelmingly conservative:
  - `168` pages at `K:M0 4b`
  - `7` pages at `K:M0 2b`
  - `1` page at `K:M2 4b`
- values were extremely tolerant:
  - all `176` value pages selected `V:M0 2b`
- fragmentation stayed modest:
  - `4` mode buckets
  - average `88.0` pages per bucket

Interpretation:

- key layers `3-21` look like the conservative core and are marked `strict`
- early key layers stay at the default `balanced`
- values can reasonably start from `aggressive`

Cheap one-token teacher-forced checks at `289` tokens were reassuring:

- exact baseline: perfect agreement
- `layer:4` key `strict`: perfect agreement
- `layer:1` key `aggressive`: perfect agreement
- default aggressive values: perfect agreement

These checks are intentionally weak, but they say the first-pass Tiny profile is at least sane.

## SmolLM2 360M

Local prefill inspection at exact `1024` prompt tokens with `balanced` K/V policy tiers showed:

- keys were genuinely mixed:
  - `287` pages at `K:M0 4b`
  - `341` pages at `K:M2 4b`
  - `12` pages at `K:M0 2b`
- values were mostly tolerant, but late layers clearly asked for rescue:
  - `581` pages at `V:M0 2b`
  - `38` pages at `V:M0 4b`
  - `21` pages at `V:M1 4b`
- fragmentation was still manageable:
  - `6` mode buckets
  - average `213.33` pages per bucket

Interpretation:

- key layers `3-5` are the conservative pocket and are marked `strict`
- late key layers `16-31` are the most approximation-tolerant and are marked `aggressive`
- late value layers `17` and `24-30` are marked `strict` because they were the first to pull `M1` or `M0 4b` under the balanced probe

Cheap one-token teacher-forced checks at `513` tokens were also reassuring:

- exact baseline: perfect agreement
- `layer:4` key `strict`: perfect agreement
- `layer:27` key `aggressive`: perfect agreement
- `layer:27` value `strict`: perfect agreement

Again, these are not enough to prove the profile is good. They are enough to justify carrying the profile forward into CUDA validation rather than starting from scratch.

## How To Use

These files are not yet wired into the benchmark CLIs directly. For now, use them as the source of:

- `--key-policy-tier`
- `--value-policy-tier`
- repeated `--key-layer-sensitivity layer:N=tier`
- repeated `--value-layer-sensitivity layer:N=tier`

The next natural step is to make the CUDA box consume these profile files directly and compare:

- exact baseline
- first-pass profile
- refined profile after loss-based validation
