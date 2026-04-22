"""Auto-generated epsilon profile (cos ≥ 0.999)."""

CALIBRATED_PROFILE = {
    4096: {
        **{lid: 0.001 for lid in range(0, 28)},
        28: 0.0005,
        **{lid: 1e-07 for lid in range(29, 32)},
    },
    8192: {
        **{lid: 0.001 for lid in range(0, 7)},
        **{lid: 0.0005 for lid in range(7, 29)},
        **{lid: 0.0001 for lid in range(29, 31)},
        31: 1e-07,
    },
    16384: {
        **{lid: 0.0005 for lid in range(0, 8)},
        8: 0.0001,
        9: 0.0005,
        **{lid: 0.0001 for lid in range(10, 29)},
        29: 1e-07,
        30: 1e-05,
        31: 1e-07,
    },
    32768: {
        **{lid: 0.0001 for lid in range(0, 28)},
        28: 5e-05,
        29: 1e-05,
        30: 0.0001,
        31: 1e-07,
    },
}
