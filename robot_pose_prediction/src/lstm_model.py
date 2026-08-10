import torch
import torch.nn as nn
import torch.nn.functional as F


class RobotPoseLSTM(nn.Module):
    """
    Predict a sequence of future robot poses from a history of robot motion.

    Input shape:
        (batch_size, history_steps, input_size)

    Default input features:
        0: relative_x
        1: relative_z
        2: relative_heading_x
        3: relative_heading_z
        4: relative_velocity_x
        5: relative_velocity_z
        6: yaw_rate

    Output shape:
        (batch_size, prediction_steps, 4)

    Output features:
        0: future_relative_x
        1: future_relative_z
        2: future_relative_heading_x
        3: future_relative_heading_z
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        prediction_steps: int = 40,
        decoder_hidden_size: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.prediction_steps = prediction_steps
        self.decoder_hidden_size = decoder_hidden_size
        self.dropout = dropout

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

        self.decoder = nn.Sequential(
            nn.Linear(
                hidden_size,
                decoder_hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                decoder_hidden_size,
                prediction_steps * 4,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:
                (B, history_steps, input_size)

        Returns:
            (B, prediction_steps, 4)
        """

        _, (hidden, _) = self.lstm(x)

        encoded = hidden[-1]

        output = self.decoder(encoded)

        output = output.view(
            x.shape[0],
            self.prediction_steps,
            4,
        )

        position = output[..., :2]
        heading = output[..., 2:4]

        heading = F.normalize(
            heading,
            p=2,
            dim=-1,
            eps=1e-8,
        )

        return torch.cat(
            [position, heading],
            dim=-1,
        )

    def get_config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "prediction_steps": self.prediction_steps,
            "decoder_hidden_size": self.decoder_hidden_size,
            "dropout": self.dropout,
        }