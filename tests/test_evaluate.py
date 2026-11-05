import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dental_3d_vae.ml.evaluate import evaluate
from dental_3d_vae.ml.model import SimpleClassifier

CPU = torch.device("cpu")


def test_loss_and_accuracy_are_exact_with_uneven_batches() -> None:
    # nn.Identity passes inputs through, so these inputs are the logits.
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0], [2.0, 0.0], [0.0, 1.0], [3.0, 0.0]])
    labels = torch.tensor([0, 1, 1, 1, 0])  # Predictions 0,1,0,1,0 -> 4 of 5 correct.
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)  # Batches: 2, 2, 1.
    loss_fn = nn.CrossEntropyLoss()

    result = evaluate(nn.Identity(), loader, loss_fn, CPU)

    assert result["accuracy"] == 0.8
    # Must equal the loss over all 5 samples at once, not the mean of batch means.
    assert result["loss"] == pytest.approx(loss_fn(logits, labels).item())


def test_evaluate_sets_eval_mode_and_leaves_weights_untouched() -> None:
    torch.manual_seed(0)
    model = SimpleClassifier(input_dim=4, num_classes=2)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    loader = DataLoader(
        TensorDataset(torch.randn(10, 4), torch.randint(0, 2, (10,))), batch_size=4
    )

    model.train()
    evaluate(model, loader, nn.CrossEntropyLoss(), CPU)

    assert not model.training
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
    assert all(param.grad is None for param in model.parameters())
