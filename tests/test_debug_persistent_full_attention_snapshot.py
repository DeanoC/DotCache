from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.debug_persistent_full_attention_snapshot import _build_markdown


def test_build_markdown_renders_prior_and_tables() -> None:
    payload = {
        "selected_block_count": 3,
        "full_block_count": 8,
        "selected_token_count": 48,
        "top_shaped_prev_attention_pages": [
            {"page_id": 0, "raw_prev_attention": 0.9, "shaped_prev_attention": 0.5},
        ],
        "top_omitted_by_priority": [
            {
                "block_id": 1,
                "token_start": 16,
                "token_count": 16,
                "region_id": 2,
                "priority_score": 1.5,
                "upper_bound": 2.5,
                "prev_attention_ema": 0.1,
                "mandatory": False,
                "soft_recent": False,
                "exploration": False,
                "optional": False,
            }
        ],
        "top_omitted_by_upper_bound": [],
        "lowest_kept_by_priority": [],
    }

    markdown = _build_markdown(payload)

    assert "Persistent Full-Attention Snapshot Debug" in markdown
    assert "Top Omitted By Priority" in markdown
    assert "0.900000" in markdown
