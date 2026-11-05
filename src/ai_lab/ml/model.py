import torch
from torch import nn

class SimpleClassifier(nn.Module):
    """A small test nn for classification"""
    def __init__(
            self,
            input_dim:int = 20,
            hidden_dim:int = 64,
            num_classes:int = 2,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)