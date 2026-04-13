#!/usr/bin/env python3

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRACKING_DIR = ROOT / "results" / "qwen35_hip_direct_tracking_20260413"
README = TRACKING_DIR / "README.md"
HISTORY_JSONL = TRACKING_DIR / "history.jsonl"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_history() -> list[dict]:
    if not HISTORY_JSONL.exists():
        return []
    entries = []
    with HISTORY_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def fmt_ms(value) -> str:
    if value is None:
        return "null"
    return f"{value:.2f}"


def fmt_delta(value) -> str:
    if value is None:
        return "null"
    return f"{value:.6f}"


def render_checkpoint(title: str, payload: dict) -> list[str]:
    return [
        f"{title}:",
        f"- generated_text: `{payload['generated_text']}`",
        f"- prompt_token_count: {payload['prompt_token_count']}",
        f"- generated_token_count: {payload['generated_token_count']}",
        f"- oracle: `{payload['oracle']}`",
        f"- oracle_device: `{payload['oracle_device']}`",
        f"- prefill_max_delta: `{fmt_delta(payload['prefill_max_delta'])}`",
        f"- decode_max_delta: `{fmt_delta(payload['decode_max_delta'])}`",
        f"- device_load_ms: `{fmt_ms(payload['device_load_ms'])}`",
        f"- device_prefill_ms: `{fmt_ms(payload['device_prefill_ms'])}`",
        f"- device_decode_ms: `{fmt_ms(payload['device_decode_ms'])}`",
        "",
    ]


def render_history(entries: list[dict]) -> list[str]:
    lines = ["Recent history:"]
    for entry in entries[-6:]:
        lines.append(
            "- "
            + f"{entry['recorded_at_utc']} "
            + f"{entry['label']} "
            + f"prefill_max_delta={fmt_delta(entry['prefill_max_delta'])} "
            + f"decode_max_delta={fmt_delta(entry['decode_max_delta'])} "
            + f"device_prefill_ms={fmt_ms(entry['device_prefill_ms'])} "
            + f"device_decode_ms={fmt_ms(entry['device_decode_ms'])}"
        )
    lines.append("")
    return lines


def main() -> None:
    short_native = load_json(TRACKING_DIR / "hip_direct_short.json")
    longer_native = load_json(TRACKING_DIR / "hip_direct_longer.json")
    short_cpu = load_json(TRACKING_DIR / "hip_direct_short_cpu_oracle.json")
    history = load_history()

    lines = [
        "# Qwen35 HIP Direct Tracking 2026-04-13",
        "",
        "Status: `hip-direct` is runnable on `hip:0`, the harness distinguishes oracle choice, and the same-device correctness drift is fixed.",
        "",
    ]
    lines += render_checkpoint("Native-device oracle short checkpoint", short_native)
    lines += render_checkpoint("Native-device oracle longer checkpoint", longer_native)
    lines += render_checkpoint("CPU oracle short checkpoint", short_cpu)
    if history:
        lines += render_history(history)
    lines += [
        "Current notes:",
        "- CPU oracle still measures cross-device/backend drift and should not be used to judge direct-HIP correctness.",
        "- Same-device tracing proved:",
        "  - direct prefill logits match native HIP exactly",
        "  - prefill cache tensors match native HIP exactly",
        "  - decode input hidden state matches native HIP exactly",
        "  - per-layer direct decode matches native HIP exactly when run from the same state",
        "  - whole-step direct decode matches native HIP exactly when run from the same state",
        "- The remaining correctness bug was not direct-executor math. It was the old direct-runner env override bundle.",
        "- Current baseline for `hip-direct` correctness work is:",
        "  - `prefill_max_delta = 0.0`",
        "  - `decode_max_delta = 0.0`",
        "  - short and longer native-device checkpoints both green",
        "",
    ]
    README.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
