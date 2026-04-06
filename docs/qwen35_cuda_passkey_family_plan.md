# Qwen3.5 CUDA Passkey Family Plan

This note defines the next named benchmark family after the first `Needle-in-a-Haystack` pack. The goal is to add a second cheap long-context retrieval suite without waiting for a full `LongBench` integration.

## Chosen Family

- RULER-style passkey retrieval

Why this family:

- cheap to wire on top of the existing prompt-builder and serving harness
- still distinct enough from the first Needle pack to count as a second named synthetic retrieval family
- exact-match should be cleaner than the earlier `shipment_token` Needle prompt because the answers are short digit strings

## Branch Components

- prompt pack: [qwen35_cuda_passkey_pack_v1.json](/Users/deanocalver/Documents/Projects/DotCache/configs/prompt_packs/qwen35_cuda_passkey_pack_v1.json)
- runner: [run_qwen35_cuda_passkey_probe.py](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_cuda_passkey_probe.py)
- protocol wrapper: [run_qwen35_cuda_passkey_pack_protocol.sh](/Users/deanocalver/Documents/Projects/DotCache/scripts/run_qwen35_cuda_passkey_pack_protocol.sh)
- summary tool: [summarize_qwen35_cuda_passkey_pack.py](/Users/deanocalver/Documents/Projects/DotCache/scripts/summarize_qwen35_cuda_passkey_pack.py)

## CUDA Command

```bash
scripts/run_qwen35_cuda_passkey_pack_protocol.sh
```

Default outputs:

- [qwen35_cuda_passkey_pack_protocol_v1.jsonl](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_passkey_pack_protocol_v1.jsonl)
- [qwen35_cuda_passkey_pack_protocol_v1_summary.md](/Users/deanocalver/Documents/Projects/DotCache/benchmarks/results/qwen35_cuda_passkey_pack_protocol_v1_summary.md)

## First Run Shape

- contexts: `32768`, `49152`
- cases: `exact`, `shortlist_base`, `shortlist_l23_ctx`
- prompts: `4`
- split: `held_out`
- lane: `systems`
- task family: `passkey_retrieval`

## Success Read

- retrieval stays correct on all or nearly all shortlist rows
- shortlist remains materially faster than exact
- exact-match is cleaner than the earlier mixed Needle `shipment_token` row

## Honest Caveat

- this is still a synthetic retrieval family, not a full `RULER` paper reproduction
- the artifact should therefore be described as a RULER-style passkey pack, not as full benchmark parity with the original suite
