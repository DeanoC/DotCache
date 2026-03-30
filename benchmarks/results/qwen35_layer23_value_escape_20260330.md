# Qwen3.5 Layer-23 Value Escape Summary (2026-03-30)

This note records the current honest read from the CUDA `layer 23` ablation work.

## Main result

`layer 23` is primarily a value-side problem on this benchmark surface.

The decisive pattern from the ablation matrix was:

- `exact_exact ~= m0_exact`
- `exact_m0 ~= m0_m0`

within each selector mode. That means `K` exactness is not doing much here, while `V` exactness is carrying the quality difference.

## Winning benchmark-only rescue

The corrected `m0_v_escape` path is the current best benchmark-only rescue:

- current shortlist lane stays intact
- `K` stays approximate
- selected `layer 23` value pages are escaped onto the exact-source `M3` path

That path materially improves over `exact_m0` and recovers a meaningful fraction of the gap back to `exact_exact`, while preserving `1.0` token agreement in the tested rows.

The current best `0.8B` operating point is now the size-gated prewarm variant:

- `execution-value-escape-prewarm` enabled
- `execution-value-escape-prewarm-min-context = 49152`

That keeps `32768` on the old path, while moving the escape-page construction for `49152+` out of first-token decode and into explicit prewarm telemetry. The missing `65536` control confirmed that this is favorable at the next context up as well:

- non-prewarmed `65536`: decode `442.67 ms/step`, mean abs `0.5386`, RMSE `0.6885`
- thresholded-prewarm `65536`: decode `421.83 ms/step`, mean abs `0.5386`, RMSE `0.6885`

## What did not win

Two narrower variants did not beat full `m0_v_escape`:

- `m0_v_escape_old`
  - too much quality regression
  - runtime savings too small/inconsistent
- `m0_v_escape_top128/256/512`
  - effectively no decode win on the shortlist rows
  - quality/runtime worse on the `layer23_full_context` rows

So the current read is that the sensitive value signal is not well captured by a simple recency cutoff or a simple rank cap.

## Telemetry now exposed

The branch now reports value-escape telemetry in a more production-shaped way:

- `execution_value_escape_source_registrations`
- `execution_value_escape_prepared_page_builds`
- `execution_value_escape_cache_hits`
- `execution_value_escape_applied_pages`

and the same deltas per decode step inside `dotcache_step_runtime_breakdown`.

That should make larger-model transfer runs easier to interpret: we can tell whether a future cost is mostly coming from one-time exact-source registration, prepared-page construction, or recurring escaped-page application.
