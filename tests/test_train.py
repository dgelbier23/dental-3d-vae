import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ai_lab.ml.data import make_dataloaders, make_synthetic_classification
from ai_lab.ml.model import SimpleClassifier
from ai_lab.ml.train import train_one_epoch

CPU = torch.device("cpu")


def test_returns_per_sample_mean_and_sets_train_mode() -> None:
    # lr=0 keeps weights fixed, so the epoch loss must equal the loss over all
    # samples at once. Batches of 4, 4, 2 check the batch-size weighting.
    torch.manual_seed(0)
    model = nn.Linear(3, 2)
    features, labels = torch.randn(10, 3), torch.randint(0, 2, (10,))
    loader = DataLoader(TensorDataset(features, labels), batch_size=4)
    loss_fn = nn.CrossEntropyLoss()

    model.eval()
    loss = train_one_epoch(model, loader, torch.optim.SGD(model.parameters(), lr=0.0), loss_fn, CPU)

    assert model.training
    assert loss == pytest.approx(loss_fn(model(features), labels).item())


def test_training_reduces_loss() -> None:
    torch.manual_seed(0)
    dataset = make_synthetic_classification(n_samples=200, input_dim=8, num_classes=2, seed=0)
    train_loader, _ = make_dataloaders(dataset, val_fraction=0.2, batch_size=32, seed=0)
    model = SimpleClassifier(input_dim=8, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses = [
        train_one_epoch(model, train_loader, optimizer, nn.CrossEntropyLoss(), CPU)
        for _ in range(5)
    ]

    assert losses[-1] < 0.5 * losses[0]
