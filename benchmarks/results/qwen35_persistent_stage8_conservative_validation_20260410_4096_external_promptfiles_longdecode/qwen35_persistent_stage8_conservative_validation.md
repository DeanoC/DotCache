# Qwen3.5 Persistent Stage 8 Conservative Validation

## Replay Summary

- snapshot count: 144
- selection changed rate: 0.979
- baseline avg max abs error: 0.002097
- stage8 avg max abs error: 0.002075
- baseline max abs error: 0.025333
- stage8 max abs error: 0.025333
- stage8 avg selected M0-metadata blocks: 98.000
- stage8 avg compression-invalid blocks: 0.000

## Serving Summary

- case count: 3
- dense avg ms/step: 75.7121
- baseline avg ms/step: 5252.3728
- stage8 avg ms/step: 5275.8021
- stage8 faster than baseline rate: 0.333
- stage8 matches baseline exact rate: 1.000
- stage8 matches dense exact rate: 0.000
- stage8 avg selected M0-metadata blocks: 4704.000
- stage8 avg dense fallback count: 0.000

## Read

Replay isolates whether compression-aware M0/M3 metadata changes ranking while still executing selected blocks exactly.
Serving checks the real persistent harness for generated-id parity, latency, and whether Stage 8 fallback/compression telemetry stays conservative.
