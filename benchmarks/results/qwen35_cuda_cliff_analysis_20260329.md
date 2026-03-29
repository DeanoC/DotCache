# Qwen3.5 CUDA 16K vs 32K Cliff Analysis

## Summary

- Baseline decode grows from `265.93 ms/step` at `16384` to `1145.16 ms/step` at `32768` (`4.31x`).
- The shortlist stays flat: `2056` selected pages at both lengths, with layer `23` fixed at `356`.
- Backend trace work terms do not grow in bytes or calls, but `score` and `mix` time do: `score 2.62x`, `mix 3.04x`, `payload_bytes 1.00x`.
- Resident/runtime memory does grow materially: `resident_bytes 1.94x`, `prepared_chunk_resident_bytes 1.82x`, `decode_reserved_bytes 1.57x`.
- The first compact grouped-decode probe is a real `32768` win on CUDA:
  - `16384` stays basically flat
  - `32768` quality lane improves `1338.65 -> 1191.86 ms/step` (`1.12x` faster)
  - shortlist size, resident bytes, and quality stay flat
  - the biggest backend movement is `mix_ms_total 387.38 -> 150.91`

## Interpretation

- The 32K cliff is unlikely to be caused by attending more shortlisted pages.
- The strongest current explanation is worse memory locality or kernel efficiency under the larger resident/prepared working set.
- The compact probe materially strengthens that explanation because it improves `32768` without changing shortlist size or quality.
- The union-wide layer-23 rescue now activates correctly at 16K, but it still does not move quality enough to explain the 32K behavior.

## Union-Wide Rescue Check

- KV `0`: rescue_applied=`True`, selected_novel=`2`, selected_old_pages=`17`, union_added_pages=`3`, union_added_mean_exact_rank=`6.0`.
- KV `1`: rescue_applied=`False`, selected_novel=`0`, selected_old_pages=`17`, union_added_pages=`5`, union_added_mean_exact_rank=`41.0`.

## Next Step

- Treat the 32K cliff as a locality/working-set issue in grouped decode, not as a shortlist-size problem.
- Reproduce the compact grouped-decode win on CUDA, then test one higher context if the box can hold it.
- Isolate why compact layout helps `mix` much more than `score` on the 32K lane.
