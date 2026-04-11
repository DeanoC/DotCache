# Immutable Tensor Path

This note describes the next loader/runtime seam after the current `model-store` work.

## Current Bottleneck

The native package path is no longer dominated by package resolution or package parsing.

On the current HIP host, representative native load traces look like:

- `0.8B`
  - `load_millis ~= 3.3s`
  - `package_resolve_millis ~= 0.18s`
  - `model_build_millis ~= 3.19s`
- `2B`
  - `load_millis ~= 4.3s`
  - `package_resolve_millis ~= 0.18s`
  - `model_build_millis ~= 4.1s`
- `4B`
  - `load_millis ~= 9.1s`
  - `package_resolve_millis ~= 0.18s`
  - `model_build_millis ~= 8.9s`

Within `model_build`, the dominant tensor is:

- `model.language_model.embed_tokens.weight`

On `0.8B`, that single tensor accounts for roughly:

- `508 MB`
- `~2.7s` of `weight_tensor_load_millis`

So the next bottleneck is backend materialization of immutable weights, not package construction.

## Goal

Introduce a first-class immutable tensor path so very large read-only weights do not have to
follow the same eager materialization model as ordinary tensors.

This is primarily about:

- `embed_tokens.weight`
- tied `lm_head.weight`

and later, if justified:

- a small set of very large projection weights

## Non-Goals

This is **not** a promise of universal zero-copy execution.

Specifically, this path does not assume:

- all Candle execution tensors can become mmap-backed
- HIP/CUDA can directly mmap file-backed blobs as device-native memory
- every immutable weight should avoid eager materialization

The point is to create a controlled alternative path for the biggest read-only tensors.

## V1 Shape

### 1. Immutable weight handle in `model-store`

Add a new readonly weight descriptor, for example:

- `ImmutableWeightHandle`

It should capture:

- package root / blob identity
- tensor name
- dtype
- shape
- layout tag
- byte range
- target backend/family the package was built for

It must be cloneable and cheap.

### 2. Provider support for handles

Extend the package/provider layer with a second access mode in addition to `get(...)`.

For example:

- `get(name) -> Tensor`
- `get_immutable(name) -> ImmutableWeightHandle`

This keeps the eager path intact while allowing selected model builders to opt into immutable
handling.

### 3. Execution-facing wrappers

Do not thread raw handles through the whole model.

Introduce specific wrappers where the win is real:

- `EmbeddingSource`
  - `Materialized(Embedding)`
  - `Immutable(ImmutableWeightHandle, hidden_size)`

- `OutputProjectionSource`
  - `Linear(Linear)`
  - `TiedImmutable(ImmutableWeightHandle)`

The first target is the embedding table and tied output projection only.

### 4. Backend behavior by target

#### CPU

Long-term goal:

- true mmap-backed readonly storage

Short-term acceptable behavior:

- keep CPU eager if the backend cannot consume readonly external storage yet

CPU is not the main reason for this work.

#### HIP / CUDA

Provide two explicit policies:

- `eager_device`
  - current behavior
  - one upload into backend storage
- `shared_host_experimental`
  - immutable tensor remains host-backed
  - execution path uses backend-specific host-visible / managed / shared-access behavior

`shared_host_experimental` should be:

- opt-in
- restricted to immutable tensors only
- backend-specific
- benchmarked independently

On UMA, this is the only path that can plausibly reduce the duplicate-residency problem for the
largest weights.

## Why not keep trying lazy materialization?

A naive lazy-load embedding path is not enough.

It tends to do one of two bad things:

- still materialize during model build because tied projections or dtype inference force it
- or simply shift the same cost to first inference without reducing total work

That is not the same as a true immutable-weight path.

The real change has to happen at the storage/consumption boundary.

## Rollout Order

### Phase 1

Add the immutable handle type and provider API.

Acceptance:

- package/provider APIs stay backward-compatible for eager users
- no model behavior changes yet

### Phase 2

Wire `embed_tokens.weight` through `EmbeddingSource`.

Acceptance:

- eager path still works unchanged
- immutable path compiles and runs behind an explicit gate
- load trace shows whether eager embedding materialization actually disappears from model-build

### Phase 3

Wire tied `lm_head` through the same immutable backing.

Acceptance:

- no duplicate eager materialization for tied output projection
- parity on existing `0.8B` smoke

### Phase 4

Benchmark `0.8B`, `2B`, `4B` for:

- total load time
- model-build time
- peak RSS
- first token latency

The immutable path is only worth keeping if it improves the intended metric instead of merely
moving cost around.

## What success looks like

For the first target (`embed_tokens.weight`), success means:

- native load trace no longer spends the same `~2.7s` inside eager embedding materialization
- peak RSS falls or at least does not regress
- first-use latency impact is understood and acceptable

If those do not happen, the immutable path should remain experimental rather than becoming the
default.

## Current Recommendation

The next implementation step should be:

1. add `ImmutableWeightHandle`
2. add provider access for immutable weights
3. prototype `EmbeddingSource::{Materialized, Immutable}`
4. keep it gated and benchmarked

That is the correct seam. More package trimming is no longer the limiting factor.
