"""CalibratedProfile: per-head, per-context-bucket epsilon + execution floor.

Produced by calibration (Phases 1-3), consumed at inference time.
Small file (~200KB for 32-layer, 32-head, 5-bucket model).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np


@dataclass
class CalibratedProfile:
    """Produced by calibration, consumed at inference."""

    # Per-head, per-context-bucket epsilon
    # Shape: [num_ctx_buckets, num_layers, num_heads]
    epsilon: np.ndarray  # float32

    # Per-layer, per-head execution floor (cos at ε=0)
    # Shape: [num_ctx_buckets, num_layers, num_heads]
    execution_floor: np.ndarray  # float32

    # Per-layer V-format policy
    v_format: list[str]  # 'int4' or 'int8' per layer

    # Context length bucket boundaries
    ctx_buckets: list[int]  # e.g. [4096, 8192, 16384, 32768, 65536]

    # Metadata
    model_name: str = ""
    calibration_date: str = ""
    num_prompts: int = 0
    target_cos: float = 0.999

    def get_epsilon(self, ctx_len: int, layer: int, head: int) -> float:
        """Look up epsilon for (ctx_len, layer, head)."""
        bucket = self._find_bucket(ctx_len)
        return float(self.epsilon[bucket, layer, head])

    def get_layer_head_epsilons(self, ctx_len: int, layer: int) -> np.ndarray:
        """Get all head epsilons for a layer. Returns [num_heads] float32."""
        bucket = self._find_bucket(ctx_len)
        return self.epsilon[bucket, layer, :]

    def get_layer_epsilons_min(self, ctx_len: int) -> dict[int, float]:
        """Get per-layer epsilon (min across heads) for backward compat."""
        bucket = self._find_bucket(ctx_len)
        num_layers = self.epsilon.shape[1]
        return {
            lid: float(self.epsilon[bucket, lid, :].min())
            for lid in range(num_layers)
        }

    def get_execution_floor(self, ctx_len: int, layer: int, head: int) -> float:
        """Get execution floor for (ctx_len, layer, head)."""
        bucket = self._find_bucket(ctx_len)
        return float(self.execution_floor[bucket, layer, head])

    def _find_bucket(self, ctx_len: int) -> int:
        """Find bucket index. Conservative: use next-larger-or-equal bucket."""
        for i, boundary in enumerate(self.ctx_buckets):
            if ctx_len <= boundary:
                return i
        return len(self.ctx_buckets) - 1

    def save(self, path: str) -> None:
        """Save to disk as compressed npz."""
        np.savez_compressed(
            path,
            epsilon=self.epsilon,
            execution_floor=self.execution_floor,
            v_format=np.array(self.v_format),
            ctx_buckets=np.array(self.ctx_buckets),
            model_name=np.array(self.model_name),
            target_cos=np.array(self.target_cos),
            calibration_date=np.array(self.calibration_date),
            num_prompts=np.array(self.num_prompts),
        )

    @classmethod
    def load(cls, path: str) -> "CalibratedProfile":
        data = np.load(path, allow_pickle=True)
        return cls(
            epsilon=data["epsilon"],
            execution_floor=data["execution_floor"],
            v_format=data["v_format"].tolist(),
            ctx_buckets=data["ctx_buckets"].tolist(),
            model_name=str(data["model_name"]),
            calibration_date=str(data.get("calibration_date", "")),
            num_prompts=int(data.get("num_prompts", 0)),
            target_cos=float(data["target_cos"]),
        )

    def summary(self) -> str:
        """Human-readable summary."""
        num_buckets, num_layers, num_heads = self.epsilon.shape
        lines = [
            f"CalibratedProfile: {self.model_name}",
            f"  Target cos: {self.target_cos}",
            f"  Buckets: {self.ctx_buckets}",
            f"  Shape: [{num_buckets} buckets, {num_layers} layers, {num_heads} heads]",
            f"  Prompts: {self.num_prompts}",
            f"  Date: {self.calibration_date}",
        ]
        for bi, ctx in enumerate(self.ctx_buckets):
            eps_slice = self.epsilon[bi]  # [layers, heads]
            floor_slice = self.execution_floor[bi]
            active = (eps_slice > 0).sum()
            total = eps_slice.size
            worst_floor = floor_slice.min()
            lines.append(
                f"  {ctx//1024:>3}K: {active}/{total} heads active, "
                f"ε range [{eps_slice[eps_slice>0].min():.0e}, {eps_slice.max():.0e}] "
                f"(floor worst={worst_floor:.4f})"
                if active > 0
                else f"  {ctx//1024:>3}K: 0/{total} heads active (floor worst={worst_floor:.4f})"
            )
        return "\n".join(lines)
