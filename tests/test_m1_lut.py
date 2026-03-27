import numpy as np

from dotcache.attention_reference import explicit_dequantized_mix, explicit_dequantized_score, mix_page_ref, score_page_ref
from dotcache.config import DotCacheConfig
from dotcache.decode_reference import decode_page
from dotcache.encode import encode_page
from dotcache.modes.m1_lut import quantize_tensor_lut


def _quantize_tensor_lut_scalar_reference(
    values: np.ndarray,
    *,
    group_size: int,
    bits: int,
    segment_count: int = 1,
    refine_steps: int = 6,
    preconditioner: str = "none",
    precondition_strength: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, int]:
    token_count, head_dim = values.shape
    num_groups = (head_dim + group_size - 1) // group_size
    padded_head_dim = num_groups * group_size
    padded = np.pad(values, ((0, 0), (0, padded_head_dim - head_dim)))
    grouped = padded.reshape(token_count, num_groups, group_size)
    levels = 1 << bits
    quantile_positions = np.linspace(0.0, 1.0, num=levels, dtype=np.float32)
    segment_slices = np.array_split(np.arange(token_count, dtype=np.int32), max(1, min(int(segment_count), token_count)))

    codebooks = np.zeros((num_groups, len(segment_slices), levels), dtype=np.float32)
    codes = np.zeros((token_count, num_groups, group_size), dtype=np.uint8)
    for group_index in range(num_groups):
        group_codes = np.zeros((token_count, group_size), dtype=np.uint8)
        for segment_index, token_indices in enumerate(segment_slices):
            segment_values = grouped[token_indices, group_index, :]
            flat_segment_values = segment_values.reshape(-1)
            fit_values = flat_segment_values
            restore_mean = 0.0
            restore_scale = 1.0
            if preconditioner == "tanh":
                restore_mean = float(np.mean(flat_segment_values, dtype=np.float64))
                centered = flat_segment_values - restore_mean
                restore_scale = float(np.std(centered, dtype=np.float64))
                if restore_scale < 1e-6:
                    restore_scale = 1.0
                fit_values = np.tanh(centered / (restore_scale * precondition_strength)).astype(np.float32)
            elif preconditioner != "none":
                raise ValueError("unsupported preconditioner")
            lut = np.quantile(fit_values, quantile_positions).astype(np.float32)
            if levels > 1:
                for _ in range(refine_steps):
                    boundaries = (lut[:-1] + lut[1:]) * 0.5
                    flat_codes = np.searchsorted(boundaries, fit_values, side="left").astype(np.int32)
                    updated = lut.copy()
                    for code_index in range(levels):
                        members = fit_values[flat_codes == code_index]
                        if members.size > 0:
                            updated[code_index] = float(np.mean(members, dtype=np.float64))
                    if np.allclose(updated, lut, atol=1e-6, rtol=0.0):
                        lut = updated
                        break
                    lut = updated
                boundaries = (lut[:-1] + lut[1:]) * 0.5
                source_values = segment_values
                if preconditioner == "tanh":
                    source_values = np.tanh(
                        (source_values - restore_mean) / (restore_scale * precondition_strength)
                    ).astype(np.float32)
                group_codes[token_indices] = np.searchsorted(boundaries, source_values, side="left").astype(np.uint8)
            if preconditioner == "tanh":
                lut = np.clip(lut, -0.999, 0.999)
                lut = (
                    np.arctanh(lut).astype(np.float32) * np.float32(restore_scale * precondition_strength)
                    + np.float32(restore_mean)
                )
            codebooks[group_index, segment_index] = lut
        codes[:, group_index] = np.clip(group_codes, 0, levels - 1)
    return codes, codebooks, padded_head_dim


def test_lut_quantization_shapes_and_error() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(4, 48)).astype(np.float32)
    codes, codebooks, padded_head_dim = quantize_tensor_lut(values, group_size=32, bits=4, segment_count=2)
    decoded = np.zeros((4, padded_head_dim), dtype=np.float32)
    segment_ids = (np.arange(codes.shape[0], dtype=np.int64) * codebooks.shape[1]) // codes.shape[0]
    for token_index in range(codes.shape[0]):
        for group_index in range(codes.shape[1]):
            start = group_index * 32
            end = start + 32
            decoded[token_index, start:end] = codebooks[group_index, segment_ids[token_index]][codes[token_index, group_index]]

    assert codes.shape == (4, 2, 32)
    assert codebooks.shape == (2, 2, 16)
    assert padded_head_dim == 64
    assert np.max(np.abs(values - decoded[:, :48])) < 0.9


def test_lut_quantization_matches_scalar_reference() -> None:
    rng = np.random.default_rng(70)
    values = rng.normal(size=(8, 48)).astype(np.float32)

    ref_codes, ref_codebooks, ref_padded = _quantize_tensor_lut_scalar_reference(
        values,
        group_size=32,
        bits=4,
        segment_count=2,
        refine_steps=4,
        preconditioner="tanh",
        precondition_strength=2.0,
    )
    fast_codes, fast_codebooks, fast_padded = quantize_tensor_lut(
        values,
        group_size=32,
        bits=4,
        segment_count=2,
        refine_steps=4,
        preconditioner="tanh",
        precondition_strength=2.0,
    )

    assert fast_padded == ref_padded
    np.testing.assert_array_equal(fast_codes, ref_codes)
    np.testing.assert_allclose(fast_codebooks, ref_codebooks, atol=1e-6, rtol=0.0)


def test_m1_page_decodes_and_matches_explicit_reference() -> None:
    rng = np.random.default_rng(8)
    config = DotCacheConfig(head_dim=48, group_size=32, bits_k=4, bits_v=4)
    keys = rng.normal(size=(6, 48)).astype(np.float32)
    values = rng.normal(size=(6, 48)).astype(np.float32)
    query = rng.normal(size=(48,)).astype(np.float32)
    attn = rng.normal(size=(6,)).astype(np.float32)
    attn = np.exp(attn - np.max(attn))
    attn = attn / np.sum(attn)

    key_page = encode_page(keys, config, kind="K", mode="M1")
    value_page = encode_page(values, config, kind="V", mode="M1")

    dense_key = decode_page(key_page)
    dense_value = decode_page(value_page)
    assert dense_key.shape == keys.shape
    assert dense_value.shape == values.shape

    np.testing.assert_allclose(score_page_ref(query, key_page), explicit_dequantized_score(query, key_page), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(mix_page_ref(attn, value_page), explicit_dequantized_mix(attn, value_page), atol=1e-5, rtol=1e-5)


def test_tanh_preconditioning_preserves_valid_lut_decode() -> None:
    rng = np.random.default_rng(9)
    core = rng.normal(scale=0.5, size=(32, 48)).astype(np.float32)
    outliers = rng.normal(loc=0.0, scale=6.0, size=(32, 48)).astype(np.float32)
    mask = rng.random(size=(32, 48)) < 0.08
    values = np.where(mask, outliers, core).astype(np.float32)

    plain_config = DotCacheConfig(head_dim=48, group_size=32, bits_k=4, default_mode_k="M1", quant_scheme_k="lut")
    conditioned_config = DotCacheConfig(
        head_dim=48,
        group_size=32,
        bits_k=4,
        default_mode_k="M1",
        quant_scheme_k="lut",
        preconditioner="tanh",
        precondition_strength=2.0,
        m1_fallback_to_m0=False,
    )

    conditioned_page = encode_page(values, conditioned_config, kind="K", mode="M1")
    plain_page = encode_page(values, plain_config, kind="K", mode="M1")

    plain_error = np.mean(np.abs(values - decode_page(plain_page)))
    conditioned_decoded = decode_page(conditioned_page)
    conditioned_error = np.mean(np.abs(values - conditioned_decoded))

    assert conditioned_page.codebooks is not None
    assert conditioned_decoded.shape == values.shape
    assert np.isfinite(conditioned_decoded).all()
    assert np.isfinite(conditioned_error)
    assert conditioned_error > 0.0
    assert plain_error > 0.0


def test_m1_can_fallback_to_m0_for_bad_pages() -> None:
    rng = np.random.default_rng(10)
    values = rng.standard_cauchy(size=(32, 48)).astype(np.float32) * np.float32(3.0)
    config = DotCacheConfig(
        head_dim=48,
        group_size=32,
        bits_k=4,
        default_mode_k="M1",
        quant_scheme_k="lut",
        m1_fallback_to_m0=True,
        m1_error_threshold=1e-6,
    )

    page = encode_page(values, config, kind="K", mode="M1")

    assert page.header.mode_default == "M0"
    assert page.scales is not None
    assert page.codebooks is None


def test_m1_can_stay_enabled_for_reasonable_pages() -> None:
    rng = np.random.default_rng(11)
    values = rng.normal(scale=0.4, size=(32, 48)).astype(np.float32)
    config = DotCacheConfig(
        head_dim=48,
        group_size=32,
        bits_k=4,
        default_mode_k="M1",
        quant_scheme_k="lut",
        m1_fallback_to_m0=True,
        m1_error_threshold=0.8,
    )

    page = encode_page(values, config, kind="K", mode="M1")

    assert page.header.mode_default == "M1"
    assert page.codebooks is not None


def test_m1_can_fallback_on_token_error_signal() -> None:
    rng = np.random.default_rng(12)
    values = rng.normal(scale=0.2, size=(32, 48)).astype(np.float32)
    values[::2, :24] += np.float32(6.0)
    values[1::2, 24:] -= np.float32(6.0)
    config = DotCacheConfig(
        head_dim=48,
        group_size=32,
        bits_k=4,
        default_mode_k="M1",
        quant_scheme_k="lut",
        m1_segment_count_k=1,
        m1_fallback_to_m0=True,
        m1_error_threshold=10.0,
        m1_token_p95_error_threshold=1e-6,
    )

    page = encode_page(values, config, kind="K", mode="M1")

    assert page.header.mode_default == "M0"
    assert page.trial_token_p95_error is not None
    assert page.trial_token_p95_error > 1e-6
