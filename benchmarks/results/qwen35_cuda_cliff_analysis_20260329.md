# Qwen3.5 CUDA 16K vs 32K Cliff Analysis

## Summary

- Baseline decode grows from `265.93 ms/step` at `16384` to `1145.16 ms/step` at `32768` (`4.31x`).
- The shortlist stays flat: `2056` selected pages at both lengths, with layer `23` fixed at `356`.
- Backend trace work terms do not grow in bytes or calls, but `score` and `mix` time do: `score 2.62x`, `mix 3.04x`, `payload_bytes 1.00x`.
- Resident/runtime memory does grow materially: `resident_bytes 1.94x`, `prepared_chunk_resident_bytes 1.82x`, `decode_reserved_bytes 1.57x`.

## Interpretation

- The 32K cliff is unlikely to be caused by attending more shortlisted pages.
- The strongest current explanation is worse memory locality or kernel efficiency under the larger resident/prepared working set.
- The union-wide layer-23 rescue now activates correctly at 16K, but it still does not move quality enough to explain the 32K behavior.

## Union-Wide Rescue Check

- KV `0`: rescue_applied=`True`, selected_novel=`2`, selected_old_pages=`17`, union_added_pages=`3`, union_added_mean_exact_rank=`6.0`.
- KV `1`: rescue_applied=`False`, selected_novel=`0`, selected_old_pages=`17`, union_added_pages=`5`, union_added_mean_exact_rank=`41.0`.

## Next Step

- Investigate the 32K cliff as a locality/working-set issue in the grouped decode backend, not as a shortlist-size problem.
