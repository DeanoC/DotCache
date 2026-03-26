from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from .config import DotCacheConfig


def _coerce_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_flat_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    data: dict[str, Any] = {}
    for line_number, raw_line in enumerate(config_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" not in raw_line:
            raise ValueError(f"invalid config line {line_number}: expected 'key: value'")
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"invalid config line {line_number}: missing key")
        data[key] = _coerce_scalar(raw_value)
    return data


def load_dotcache_config(path: str | Path) -> DotCacheConfig:
    raw = load_flat_yaml(path)
    valid_keys = {field.name for field in fields(DotCacheConfig)}
    unknown_keys = sorted(set(raw) - valid_keys)
    if unknown_keys:
        raise ValueError(f"unknown DotCacheConfig fields in {path}: {', '.join(unknown_keys)}")
    return DotCacheConfig(**raw)
