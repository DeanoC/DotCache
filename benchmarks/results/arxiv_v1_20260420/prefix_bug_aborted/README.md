# Aborted sweep — pre-trailing-fix hybrid path

The first Paper-1 hybrid re-sweep (commit `ac87e753`) completed cell 04
(pg19 4K certified) before we caught the trailing-partial-block mass
leak in the hybrid kernel — positions between `num_tokens` and the next
block boundary contributed zero-score mass to the softmax, biasing
perplexity by ~+0.15 vs dense.

The 04 result here (ppl 6.989, Δppl +0.151) is a snapshot of that bug
in action. Kept only as a debugging reference. The correct Paper-1 run
is under the main directory after commit `308357bd`
(last_block_valid kernel parameter).

The 05_niah_4K_certified.log was an orphan stdout stream from the
aborted sweep's cell 05 (which was killed mid-run before producing a
JSON).
