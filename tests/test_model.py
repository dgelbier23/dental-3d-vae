import torch
from ai_lab.ml.model import SimpleClassifier

def test_classifier_output_shape():
    model = SimpleClassifier(input_dim=20, num_classes=3)
    inputs = torch.randn(8, 20)

    outputs = model(inputs)

    assert outputs.shape == (8, 3)

def test_classifier_outputs_finite_values():
    model = SimpleClassifier()
    outputs = model(torch.randn(4, 20))

    assert torch.isfinite(outputs).all()