import numpy as np

from dotcache.modes.m0_affine import dequantize_groups, quantize_tensor


def test_affine_quantization_shapes_and_error() -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(size=(4, 48)).astype(np.float32)
    codes, scales, bias, padded_head_dim = quantize_tensor(values, group_size=32, bits=4, scheme="affine")
    decoded = dequantize_groups(codes, scales=scales, bias=bias, bits=4, scheme="affine").reshape(4, padded_head_dim)[:, :48]

    assert codes.shape == (4, 2, 32)
    assert scales.shape == (4, 2)
    assert bias is not None
    assert bias.shape == (4, 2)
    assert padded_head_dim == 64
    assert np.max(np.abs(values - decoded)) < 0.35


def test_symmetric_quantization_roundtrip_is_reasonable() -> None:
    rng = np.random.default_rng(1)
    values = rng.normal(size=(3, 32)).astype(np.float32)
    codes, scales, bias, padded_head_dim = quantize_tensor(values, group_size=32, bits=4, scheme="symmetric")
    decoded = dequantize_groups(codes, scales=scales, bias=bias, bits=4, scheme="symmetric").reshape(3, padded_head_dim)

    assert bias is None
    assert np.max(np.abs(values - decoded)) < 0.4

