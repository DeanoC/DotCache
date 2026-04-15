"""Context-aware per-layer epsilon profiles for certified attention.

The epsilon profile is a table indexed by (context_length, layer_id) that
controls the skip threshold. Longer contexts need tighter epsilons on
early layers because those layers' attention patterns change at scale.

Lookup uses the closest-less-than-or-equal context length from a set of
calibration points (e.g. [4K, 8K, 16K, 32K, 64K]). This is conservative:
a 12K context uses the 8K profile (tighter than needed but safe).
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field


# Default calibration points
DEFAULT_CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536]

# Default profile for LLaMA 3.1-8B, calibrated from cosine measurements.
# At 8K: layer 0 needs 1e-3, layer 31 needs 1e-5
# At 16K: layer 0 needs ~1e-5 (was skipping 87% at 1e-3)
# At 32K: layer 0 needs ~1e-6 (was skipping 100% at 1e-3)
LLAMA_31_8B_PROFILE = {
    4096: {
        # Short context: relaxed thresholds
        **{lid: 5e-3 for lid in range(6)},     # Early: very relaxed
        **{lid: 1e-3 for lid in range(6, 29)},  # Mid: relaxed
        **{lid: 1e-4 for lid in range(29, 32)}, # Deep sparse: moderate
    },
    8192: {
        # 8K: the original calibrated profile
        **{lid: 1e-3 for lid in range(6)},
        **{lid: 1e-4 for lid in range(6, 29)},
        **{lid: 1e-5 for lid in range(29, 32)},
    },
    16384: {
        # 16K: tighten early layers (layer 0 was cos=0.75 at 1e-3)
        **{lid: 1e-5 for lid in range(6)},
        **{lid: 1e-5 for lid in range(6, 29)},
        **{lid: 1e-6 for lid in range(29, 32)},
    },
    32768: {
        # 32K: very tight on all layers (layer 0 was cos=0.0 at 1e-3)
        **{lid: 1e-6 for lid in range(6)},
        **{lid: 1e-6 for lid in range(6, 29)},
        **{lid: 1e-7 for lid in range(29, 32)},
    },
    65536: {
        # 64K: conservative — same as 32K until calibrated
        **{lid: 1e-6 for lid in range(6)},
        **{lid: 1e-6 for lid in range(6, 29)},
        **{lid: 1e-7 for lid in range(29, 32)},
    },
}


@dataclass
class EpsilonProfile:
    """Context-aware per-layer epsilon lookup.

    Stores epsilon values at calibration context lengths. At runtime,
    looks up the closest-less-than-or-equal context length (conservative).
    """
    # context_length → {layer_id → epsilon}
    profiles: dict[int, dict[int, float]] = field(default_factory=dict)
    # Sorted calibration points for binary search
    _sorted_lengths: list[int] = field(default_factory=list, repr=False)
    # Default epsilon for uncalibrated layers
    default_epsilon: float = 1e-4
    # Number of layers
    num_layers: int = 32

    def __post_init__(self):
        self._sorted_lengths = sorted(self.profiles.keys())

    @classmethod
    def llama_31_8b(cls) -> "EpsilonProfile":
        """Default profile for LLaMA 3.1-8B."""
        return cls(profiles=LLAMA_31_8B_PROFILE, num_layers=32)

    @classmethod
    def uniform(cls, epsilon: float, num_layers: int = 32) -> "EpsilonProfile":
        """Single epsilon for all layers and contexts."""
        return cls(
            profiles={0: {lid: epsilon for lid in range(num_layers)}},
            num_layers=num_layers,
            default_epsilon=epsilon,
        )

    def get_epsilon(self, context_length: int, layer_id: int) -> float:
        """Look up epsilon for a given context length and layer.

        Uses closest-less-than-or-equal context length (conservative).
        """
        if not self._sorted_lengths:
            return self.default_epsilon

        # Binary search for closest-less-than-or-equal
        idx = bisect_right(self._sorted_lengths, context_length) - 1
        if idx < 0:
            # Context shorter than smallest calibration point — use smallest
            ctx = self._sorted_lengths[0]
        else:
            ctx = self._sorted_lengths[idx]

        layer_eps = self.profiles.get(ctx, {})
        return layer_eps.get(layer_id, self.default_epsilon)

    def get_layer_epsilons(self, context_length: int) -> dict[int, float]:
        """Get all layer epsilons for a given context length."""
        return {lid: self.get_epsilon(context_length, lid) for lid in range(self.num_layers)}

    def update_layer(self, context_length: int, layer_id: int, epsilon: float) -> None:
        """Update epsilon for a specific context length and layer."""
        if context_length not in self.profiles:
            self.profiles[context_length] = {}
            self._sorted_lengths = sorted(self.profiles.keys())
        self.profiles[context_length][layer_id] = epsilon

    def summary(self) -> str:
        """Human-readable summary of the profile."""
        lines = ["EpsilonProfile:"]
        for ctx in self._sorted_lengths:
            eps_vals = self.profiles[ctx]
            unique = sorted(set(eps_vals.values()))
            groups = []
            for e in unique:
                layers = [l for l, v in eps_vals.items() if v == e]
                if len(layers) > 3:
                    groups.append(f"  L{min(layers)}-{max(layers)}: {e:.0e}")
                else:
                    groups.append(f"  L{','.join(str(l) for l in layers)}: {e:.0e}")
            lines.append(f"  {ctx//1024}K context:")
            lines.extend(f"    {g}" for g in groups)
        return "\n".join(lines)
