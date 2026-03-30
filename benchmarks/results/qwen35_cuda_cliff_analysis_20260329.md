# Qwen3.5 CUDA 16K vs 32K Cliff Analysis

## Summary

- Baseline decode grows from `265.93 ms/step` at `16384` to `1145.16 ms/step` at `32768` (`4.31x`).
- The shortlist stays flat: `2056` selected pages at both lengths, with layer `23` fixed at `356`.
- Backend trace work terms do not grow in bytes or calls, but `score` and `mix` time do: `score 2.62x`, `mix 3.04x`, `payload_bytes 1.00x`.
- Resident/runtime memory does grow materially: `resident_bytes 1.94x`, `prepared_chunk_resident_bytes 1.82x`, `decode_reserved_bytes 1.57x`.
- The compact grouped-decode probe touched the right area, but did not reproduce as a clean stable win:
  - one earlier `32768` run improved materially
  - the fresh confirm run is mixed across `32768` and `49152`
  - shortlist size, resident bytes, and quality stayed flat throughout
  - the main remaining signal is still that `mix` is unusually sensitive to layout

## Interpretation

- The 32K cliff is unlikely to be caused by attending more shortlisted pages.
- The strongest current explanation is worse memory locality or kernel efficiency under the larger resident/prepared working set.
- The compact probe still supports that explanation directionally, but not strongly enough yet to treat compaction itself as the answer.
- The union-wide layer-23 rescue now activates correctly at 16K, but it still does not move quality enough to explain the 32K behavior.

## Union-Wide Rescue Check

- KV `0`: rescue_applied=`True`, selected_novel=`2`, selected_old_pages=`17`, union_added_pages=`3`, union_added_mean_exact_rank=`6.0`.
- KV `1`: rescue_applied=`False`, selected_novel=`0`, selected_old_pages=`17`, union_added_pages=`5`, union_added_mean_exact_rank=`41.0`.

## Next Step

- Treat the 32K cliff as a locality/working-set issue in grouped decode, not as a shortlist-size problem.
- Stop treating full compaction as the likely final answer.
- Isolate narrower grouped-`mix` layout changes first.
- Keep the shortlist baseline as the main lane while probing locality variants benchmark-only.
