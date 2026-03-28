# Local Layer Profiles

These are first-pass handwritten layer sensitivity profiles derived from local Apple MPS inspection runs. They are meant to be useful hints while the CUDA box is busy, not final production policies.

The two supporting tools are:

- [bench_layer_sensitivity.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_layer_sensitivity.py)
- [inspect_policy_prefill.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/inspect_policy_prefill.py)

The current profile artifacts live in:

- [tinyllama_local_first_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/tinyllama_local_first_pass.yaml)
- [smollm2_360m_local_first_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/smollm2_360m_local_first_pass.yaml)
- [smollm2_360m_local_second_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/smollm2_360m_local_second_pass.yaml)
- [smollm2_360m_local_third_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/smollm2_360m_local_third_pass.yaml)

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

### SmolLM2 Second Pass

The first-pass SmolLM2 profile was useful as a failure case:

- it improved KV memory
- but it made late keys too `M2`-heavy
- and, more importantly, it made late values far too `M1`-heavy

The key implementation detail is that local `strict` values are **not** “more exact” in the current planner. For values, `strict` prefers:

- `V:M1 4b`
- then `V:M0 4b`

That means the first-pass late-value `strict` overrides actually increased `M1` usage and blew up teacher-forced loss on SmolLM2.

The second-pass SmolLM2 profile therefore pulls back to a much safer shape:

- keep only the early key strict pocket at layers `3-5`
- remove all late key aggressive overrides
- remove all value-layer overrides
- use default `balanced` policy for the rest

That second pass is the more realistic CUDA hint:

- the early key pocket still looks meaningfully strict
- the default balanced policy already finds approximate opportunities without forcing them
- values should be allowed to stay mostly `M0 2b` unless we have stronger evidence that `M1` is safe on that layer

### SmolLM2 Third Pass

The third pass asked a narrower question: after fixing the bad value-side behavior, is the remaining loss mostly caused by late key `M2` usage?

To test that, the third-pass profile:

- kept the early strict key pocket at layers `3-5`
- kept values at the safer default `balanced`
- additionally marked the deepest key layers `24-31` as `strict`

That changed the local exact `1024` prompt mix from:

- second pass:
  - `K:M2 4b`: `341`
  - `K:M0 4b`: `287`

to:

- third pass:
  - `K:M2 4b`: `223`
  - `K:M0 4b`: `405`

And on the short teacher-forced check at `1032 / 1024 / 8`:

- second pass:
  - loss delta `+0.0541`
  - token agreement `1.0`
- third pass:
  - loss delta `+0.0516`
  - token agreement `1.0`

So the third pass is a small step in the right direction, but not a breakthrough:

- it confirms that reducing late-key `M2` pressure helps
- it does **not** reduce the loss enough to make the current SmolLM2 adaptive policy look good
- the main local conclusion remains that TinyLlama is the clean success case, while SmolLM2 still needs more conservative key-side planning or better key codecs

## How To Use

These files are now wired into the benchmark CLIs directly through `--layer-profile`.

The next natural step is to make the CUDA box compare:

- exact baseline
- first-pass profile
- refined profile after loss-based validation
