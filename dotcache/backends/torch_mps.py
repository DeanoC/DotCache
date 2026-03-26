from __future__ import annotations


def mps_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.backends.mps.is_available())


def score_page_mps(*args, **kwargs):  # type: ignore[no-untyped-def]
    del args
    del kwargs
    raise NotImplementedError("torch_mps execution is the next implementation stage")


def mix_page_mps(*args, **kwargs):  # type: ignore[no-untyped-def]
    del args
    del kwargs
    raise NotImplementedError("torch_mps execution is the next implementation stage")

