import os
import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from ai_lab.runtime.device import get_device, set_seed


def _draw_random_values() -> tuple[float, float, float]:
    """Draw one value from each RNG that set_seed controls."""
    return random.random(), float(np.random.rand()), float(torch.rand(1))


@pytest.fixture(autouse=True)
def restore_torch_determinism(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo process-wide determinism settings so tests stay independent."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    saved = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.deterministic,
        torch.backends.cudnn.benchmark,
    )
    yield
    torch.use_deterministic_algorithms(saved[0])
    torch.backends.cudnn.deterministic = saved[1]
    torch.backends.cudnn.benchmark = saved[2]


# --- set_seed: reproducibility -------------------------------------------------


def test_set_seed_is_reproducible() -> None:
    set_seed(123)
    first = _draw_random_values()
    set_seed(123)
    second = _draw_random_values()

    assert first == second


def test_different_seeds_produce_different_values() -> None:
    set_seed(1)
    first = _draw_random_values()
    set_seed(2)
    second = _draw_random_values()

    # Compare each library separately so one RNG can't hide another's failure.
    assert all(a != b for a, b in zip(first, second))


# --- set_seed: input validation -------------------------------------------------


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_set_seed_accepts_boundary_values(seed: int) -> None:
    set_seed(seed)


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_set_seed_rejects_out_of_range_values(seed: int) -> None:
    with pytest.raises(ValueError):
        set_seed(seed)


# --- set_seed: deterministic mode -----------------------------------------------


def test_default_does_not_enable_deterministic_mode() -> None:
    set_seed(0)

    assert not torch.are_deterministic_algorithms_enabled()


def test_deterministic_mode_sets_all_flags() -> None:
    set_seed(0, deterministic=True)

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_deterministic_mode_keeps_existing_cublas_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    set_seed(0, deterministic=True)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"


# --- get_device -------------------------------------------------------------------


def test_get_device_returns_torch_device() -> None:
    assert isinstance(get_device(), torch.device)


def test_get_device_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert get_device() == torch.device("cuda")


def test_get_device_uses_mps_when_no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)

    assert get_device() == torch.device("mps")


def test_get_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert get_device() == torch.device("cpu")
