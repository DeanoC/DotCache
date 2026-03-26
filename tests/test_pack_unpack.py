import numpy as np

from dotcache.packing import pack_bits, unpack_bits, words_per_group


def test_pack_unpack_roundtrip_4bit() -> None:
    codes = np.array([[0, 1, 2, 3, 4, 5, 6, 15]], dtype=np.uint8)
    packed = pack_bits(codes, bits=4)
    unpacked = unpack_bits(packed, bits=4, group_size=codes.shape[-1])
    np.testing.assert_array_equal(unpacked, codes)


def test_pack_unpack_roundtrip_2bit() -> None:
    codes = np.array([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=np.uint8)
    packed = pack_bits(codes, bits=2)
    unpacked = unpack_bits(packed, bits=2, group_size=codes.shape[-1])
    np.testing.assert_array_equal(unpacked, codes)


def test_pack_unpack_handles_partial_word() -> None:
    codes = np.arange(24, dtype=np.uint8)[None, :] % 4
    packed = pack_bits(codes, bits=2)
    assert packed.shape[-1] == words_per_group(24, 2)
    unpacked = unpack_bits(packed, bits=2, group_size=24)
    np.testing.assert_array_equal(unpacked, codes)

