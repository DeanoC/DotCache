from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

from .config import DotCacheConfig


def _coerce_scalar(raw_value: str) -> Any:
    value = raw_value.strip()
    if value == "[]":
        return []
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


def load_layer_profile(path: str | Path) -> dict[str, Any]:
    profile_path = Path(path)
    lines = profile_path.read_text(encoding="utf-8").splitlines()
    data: dict[str, Any] = {}
    index = 0
    while index < len(lines):
        raw_line = lines[index]
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            index += 1
            continue
        if raw_line[:1].isspace():
            raise ValueError(f"invalid profile line {index + 1}: unexpected indentation at top level")
        if ":" not in raw_line:
            raise ValueError(f"invalid profile line {index + 1}: expected 'key: value'")
        key, raw_value = raw_line.split(":", 1)
        key = key.strip()
        value = raw_value.strip()
        if value:
            if value == ">":
                folded: list[str] = []
                index += 1
                while index < len(lines):
                    continuation = lines[index]
                    if continuation.strip() and not continuation[:1].isspace():
                        break
                    if continuation.strip():
                        folded.append(continuation.strip())
                    index += 1
                data[key] = " ".join(folded)
                continue
            data[key] = _coerce_scalar(value)
            index += 1
            continue

        block_lines: list[str] = []
        index += 1
        while index < len(lines):
            continuation = lines[index]
            if continuation.strip() and not continuation[:1].isspace():
                break
            if continuation.strip():
                block_lines.append(continuation)
            index += 1
        if not block_lines:
            data[key] = {}
            continue
        stripped_block = [line.lstrip() for line in block_lines]
        if all(line.startswith("- ") for line in stripped_block):
            data[key] = [_coerce_scalar(line[2:]) for line in stripped_block]
            continue
        nested: dict[str, Any] = {}
        block_index = 0
        while block_index < len(block_lines):
            nested_line = stripped_block[block_index]
            if ":" not in nested_line:
                raise ValueError(f"invalid nested profile entry under '{key}': expected 'child: value'")
            nested_key, nested_value_raw = nested_line.split(":", 1)
            nested_key = nested_key.strip()
            nested_value = nested_value_raw.strip()
            if nested_value == ">":
                folded: list[str] = []
                block_index += 1
                while block_index < len(block_lines):
                    candidate = block_lines[block_index]
                    if len(candidate) - len(candidate.lstrip()) <= len(block_lines[0]) - len(block_lines[0].lstrip()):
                        break
                    if candidate.strip():
                        folded.append(candidate.strip())
                    block_index += 1
                nested[nested_key] = " ".join(folded)
                continue
            nested[nested_key] = _coerce_scalar(nested_value)
            block_index += 1
        data[key] = nested
    return data


def load_dotcache_config(path: str | Path) -> DotCacheConfig:
    raw = load_flat_yaml(path)
    valid_keys = {field.name for field in fields(DotCacheConfig)}
    unknown_keys = sorted(set(raw) - valid_keys)
    if unknown_keys:
        raise ValueError(f"unknown DotCacheConfig fields in {path}: {', '.join(unknown_keys)}")
    return DotCacheConfig(**raw)
