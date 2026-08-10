import torch
import torch.nn as nn
from tabm import TabM


class QuestHandIntentEstTabM(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int = 1,
        k: int = 32,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.model = TabM.make(
            n_num_features=input_size,
            d_out=output_size,
            k=k,
            arch_type="tabm",
        )

        self.k = self.model.k

    def forward(self, x):
        """
        Returns raw logits with shape:

            [batch_size, k, output_size]
        """
        return self.model(x)