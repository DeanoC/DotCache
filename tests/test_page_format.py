from dotcache.config import DotCacheConfig
from dotcache.encode import encode_page
from dotcache.page_format import deserialize_header, serialize_header
from dotcache.types import PageHeader


def test_header_serialization_roundtrip() -> None:
    header = PageHeader(
        layer_id=1,
        kv_head_id=2,
        kind="K",
        token_start=64,
        token_count=8,
        head_dim=48,
        padded_head_dim=64,
        group_size=32,
        num_groups=2,
        bits=4,
        words_per_group=4,
        mode_default="M0",
        layout="group_major",
        quant_scheme="affine",
    )

    payload = serialize_header(header)
    decoded = deserialize_header(payload)
    assert decoded == header


def test_encoded_page_reports_payload_and_metadata_bytes() -> None:
    config = DotCacheConfig(head_dim=32)
    page = encode_page([[1.0] * 32, [2.0] * 32], config, kind="K")
    assert page.payload_nbytes > 0
    assert page.metadata_nbytes > 0
    assert page.total_nbytes == page.payload_nbytes + page.metadata_nbytes

