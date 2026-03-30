# Qwen3.5 Value Escape Transfer Summary (2026-03-30)

This note records the current cross-model read from the benchmark-only selected-page `V` escape work.

## Main result

The mechanism transfers across models, but the sensitive layer does not.

What transferred:

- selected-page `V` escape can materially improve quality without destabilizing decode
- the escape path behaves like an up-front setup plus step-0 build cost, then mostly cache-hit reuse

What did not transfer:

- the exact fragile layer
- the assumption that `layer 23` remains special on larger Qwen3.5 variants

## 0.8B result

On `Qwen/Qwen3.5-0.8B`, `layer 23` was the useful value-side rescue target on the tested CUDA shortlist path.

That path:

- improved materially over `exact_m0`
- recovered a meaningful fraction of the gap back to `exact_exact`
- kept token agreement at `1.0`

## 4B result

On `Qwen/Qwen3.5-4B`, a direct transfer of the `layer 23` rescue did not help. But a cheap value-escape layer scan showed the mechanism still works on a different layer.

At `16384`, the best candidate looked like `layer 19`, with `layer 7` second.

At `32768`, the clearer winner was `layer 7`:

- `approx_shortlist`, `layer 7`
  - mean abs `0.4656 -> 0.3654`
  - RMSE `0.6312 -> 0.4738`
  - decode `679.59 -> 670.89 ms/step`

while `layer 19` was slightly harmful at the same context.

The next scheduling check also showed that the `0.8B` prewarm policy does not transfer automatically:

- `4B @ 49152`, `layer 7`, baseline:
  - decode `831.54 ms/step`
  - mean abs `0.2790`
  - RMSE `0.3587`
- `4B @ 49152`, `layer 7`, thresholded prewarm:
  - decode `859.84 ms/step`
  - same quality

The missing `65536` control closed the loop the same way:

- `4B @ 65536`, `layer 7`, baseline:
  - decode `1024.55 ms/step`
  - mean abs `0.2712`
  - RMSE `0.3413`
- `4B @ 65536`, `layer 7`, thresholded prewarm:
  - decode `1048.77 ms/step`
  - same quality

So the mechanism transfers, but the prewarm rule should be treated as model-specific as well. The current benchmark read is:

- `0.8B`: use the value-escape mechanism with thresholded prewarm at `49152+`
- `4B`: use the value-escape mechanism on `layer 7`, but keep prewarm off

The promoted reference runner for those two benchmark-only lanes now lives at
[run_qwen35_value_escape_reference.py](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_value_escape_reference.py),
with presets:

- `qwen35_0p8b_best`
- `qwen35_4b_best`

The promoted scan entrypoint now also has named presets in
[run_qwen35_value_escape_layer_scan.py](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_value_escape_layer_scan.py):

- `qwen35_4b_initial_scan`
- `qwen35_4b_confirm_32768`

So the canonical workflow is now:

1. run a scan preset to localize the fragile value-sensitive layer
2. run the matching reference preset to benchmark the best lane cleanly

## Current strategy

The repo should now treat value escape as a reusable tuning pattern rather than a fixed-layer rule:

1. scan the candidate full-attention layers for the target model/context
2. identify the fragile value-sensitive layer
3. apply the same selected-page `V` escape mechanism there
4. decide prewarm separately for that model/layer/context regime

That is a stronger systems story than “layer 23 is magic,” because it explains both the successful `0.8B` rescue and the successful `4B` transfer without pretending the layer choice is universal.
