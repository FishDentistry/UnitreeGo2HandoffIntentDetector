import torch
import torch.nn as nn


class HandIntentMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.network = nn.Sequential(
            #nn.LayerNorm(input_size),

            nn.Linear(input_size, 256),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.25),

            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.20),

            nn.Linear(64, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

