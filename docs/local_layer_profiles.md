# Local Layer Profiles

These are first-pass handwritten layer sensitivity profiles derived from local Apple MPS inspection runs. They are meant to be useful hints while the CUDA box is busy, not final production policies.

The two supporting tools are:

- [bench_layer_sensitivity.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/bench_layer_sensitivity.py)
- [inspect_policy_prefill.py](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/inspect_policy_prefill.py)

The current profile artifacts live in:

- [tinyllama_local_first_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/tinyllama_local_first_pass.yaml)
- [tinyllama_local_second_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/tinyllama_local_second_pass.yaml)
- [tinyllama_cuda_start.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/tinyllama_cuda_start.yaml)
- [qwen35_0p8b_attention_subset_first_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/qwen35_0p8b_attention_subset_first_pass.yaml)
- [qwen35_0p8b_attention_subset_second_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/qwen35_0p8b_attention_subset_second_pass.yaml)
- [smollm2_360m_local_first_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/smollm2_360m_local_first_pass.yaml)
- [smollm2_360m_local_second_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/smollm2_360m_local_second_pass.yaml)
- [smollm2_360m_local_third_pass.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/smollm2_360m_local_third_pass.yaml)
- [smollm2_360m_cuda_start.yaml](/Users/deanocalver/Documents/Projects/DotCache/configs/layer_profiles/smollm2_360m_cuda_start.yaml)

The CUDA handoff note for these profiles is:

- [cuda_next_steps.md](/Users/deanocalver/Documents/Projects/DotCache/docs/cuda_next_steps.md)

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

The CUDA starter profile for TinyLlama keeps this exact shape. It is still the best local adaptive checkpoint, and the next CUDA probes should compare it against:

- exact `M0`
- exact `K=4b, V=3b`
- the same profile with recent-page `M3 int8`

### TinyLlama Second Pass

The second Tiny pass tested whether the earliest key layers could be made more aggressive while keeping the same long strict middle-layer key spine and the same aggressive value policy.

That profile added:

- `layer:0=aggressive`
- `layer:1=aggressive`
- `layer:2=aggressive`

while keeping `layers 3-21` key `strict`.

What happened locally:

- exact `577` prompt:
  - first pass:
    - decode `6601.72 ms/step`
    - resident KV `9.35 MB`
    - `K:M2` pages `1`
  - second pass:
    - decode `9427.54 ms/step`
    - resident KV `9.33 MB`
    - `K:M2` pages `3`

- teacher-forced `320 / 288 / 16`:
  - first pass loss delta: `+0.00008`
  - second pass loss delta: `+0.00015`
  - token agreement stayed `1.0`

So the second Tiny pass is a useful negative result:

- it did not buy a meaningful extra KV-memory win
- it increased prefill/decode cost on this Mac
- it did not improve the already-good loss behavior

That means the **first-pass Tiny profile remains the better local checkpoint**. The value-side aggression looks useful, but pushing the earliest key layers harder does not seem worth it.

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

That is why the CUDA starter profile is based on the safer second/third-pass family:

- balanced values
- an early strict key pocket
- deepest key layers clamped back to strict
- no forced late-value overrides

Two additional local hints should also carry into CUDA:

- `M0 3b` is now a plausible new intermediate tier, especially as `K=4b, V=3b`
- recent-page `M3 int8` is now planner-selectable end to end and cuts recent-page residency materially on this Mac

## How To Use

These files are now wired into the benchmark CLIs directly through `--layer-profile`.

The next natural step is to make the CUDA box compare:

- exact baseline
- first-pass profile
- refined profile after loss-based validation

## Qwen3.5 Attention Subset

Qwen3.5 is different from the TinyLlama and SmolLM2 profiles because the local policy work only applies to the six `full_attention` layers in the text stack:

- `3`
- `7`
- `11`
- `15`
- `19`
- `23`

The first useful local ablation was on the attention-subset DotCache replay lane at exact `32` prompt tokens with `tokens_per_page=16`. That ablation compared `K-only`, `V-only`, and `K+V` prefill quantization for those six layers.

The current first-pass profile is:

- default `balanced` for both keys and values
- `recent_window: 0` so the probe actually exercises sealed static pages instead of hiding everything in the live-tail `M3` path
- stricter key layers:
  - `7`
  - `11`
  - `19`
- stricter value layers:
  - `15`
  - `23`
- layer `3` left at the default because it looked mixed in the local probe

Interpretation:

- the Qwen3.5 attention-subset drift is not one-sided
- later layer `23` looked more value-sensitive
- later layer `19` looked more key-sensitive
- the first policy pass should therefore be layer-aware instead of globally “safer keys” or globally “safer values”

This is intentionally only an attention-subset profile. It does **not** say anything yet about how the DeltaNet / `linear_attention` state should be cached or compressed.

### Qwen3.5 Attention Subset Second Pass

The first pass turned out to be a useful negative result once it was rerun with `recent_window: 0` so the benchmark actually exercised sealed static pages:

- the profile did apply
- but generic late-value `strict` was the wrong lever
- on the current planner, `strict` values prefer `V:M1 4b` before `V:M0 4b`

That pushed the late attention layers too far:

- first pass, exact `32` prompt, `tokens_per_page=16`:
  - `V:M1 4b` pages: `14`
  - replay context max abs error: `0.9731`
  - teacher-forced logit max abs error: `1.8027`

The second pass keeps the same key-side strict hints, but replaces the fragile late-value tiering with explicit `M0`-first overrides at layers `15`, `19`, and `23`:

- `layer:15=M0/affine/4,M0/affine/3,M1/lut/4`
- `layer:19=M0/affine/4,M0/affine/3,M1/lut/4`
- `layer:23=M0/affine/4,M0/affine/3,M1/lut/4`

That materially improved the local exact `32` prompt result:

- second pass:
  - `V:M1 4b` pages: `4`
  - replay context max abs error: `0.2688`
  - teacher-forced logit max abs error: `1.2402`

So the current local Qwen3.5 attention-subset read is:

- layer-aware policy really does matter
- the value side needs explicit safer candidate sets, not generic `strict`
- the second-pass profile is the better local checkpoint
- the attention-subset lane is still not a performance win, but it is now a much cleaner design spike for the CUDA side and for later hybrid-state work
