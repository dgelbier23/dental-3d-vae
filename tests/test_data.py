import torch

from ai_lab.ml.data import make_dataloaders, make_synthetic_classification


def _dataset(seed: int = 0, n_samples: int = 200, num_classes: int = 3):
    return make_synthetic_classification(
        n_samples=n_samples, input_dim=8, num_classes=num_classes, seed=seed
    )


def _loaders(seed: int = 0, batch_size: int = 16):
    return make_dataloaders(_dataset(), val_fraction=0.25, batch_size=batch_size, seed=seed)


# --- make_synthetic_classification ----------------------------------------------


def test_synthetic_shapes_and_dtypes() -> None:
    features, labels = _dataset(n_samples=50).tensors

    assert features.shape == (50, 8)
    assert labels.shape == (50,)
    assert features.dtype == torch.float32
    assert labels.dtype == torch.int64  # required by nn.CrossEntropyLoss


def test_synthetic_labels_cover_all_classes() -> None:
    _, labels = _dataset(n_samples=1000, num_classes=3).tensors

    assert set(labels.tolist()) == {0, 1, 2}


def test_synthetic_depends_only_on_seed() -> None:
    torch.manual_seed(0)  # Global RNG state must not matter.
    a = _dataset(seed=1).tensors[0]
    torch.manual_seed(999)
    b = _dataset(seed=1).tensors[0]
    c = _dataset(seed=2).tensors[0]

    assert torch.equal(a, b)
    assert not torch.equal(a, c)


# --- make_dataloaders -----------------------------------------------------------


def test_split_sizes_and_no_overlap() -> None:
    train_loader, val_loader = _loaders()
    train_idx = set(train_loader.dataset.indices)
    val_idx = set(val_loader.dataset.indices)

    assert (len(train_idx), len(val_idx)) == (150, 50)
    assert train_idx.isdisjoint(val_idx)


def test_split_is_reproducible() -> None:
    def val_indices(seed: int) -> list[int]:
        return list(_loaders(seed)[1].dataset.indices)

    assert val_indices(3) == val_indices(3)
    assert val_indices(3) != val_indices(4)


def test_train_shuffle_is_reproducible_and_changes_each_epoch() -> None:
    def first_two_epochs(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
        train_loader, _ = _loaders(seed, batch_size=150)  # One batch per epoch.
        return next(iter(train_loader))[1], next(iter(train_loader))[1]

    run_a = first_two_epochs(5)
    run_b = first_two_epochs(5)

    assert all(torch.equal(a, b) for a, b in zip(run_a, run_b))
    assert not torch.equal(run_a[0], run_a[1])


def test_val_loader_is_not_shuffled() -> None:
    _, val_loader = _loaders()

    first = torch.cat([features for features, _ in val_loader])
    second = torch.cat([features for features, _ in val_loader])

    assert torch.equal(first, second)
