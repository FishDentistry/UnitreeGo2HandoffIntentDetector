import math
from typing import Dict

import torch
import torch.nn as nn


class RobotPoseMDNLSTM(nn.Module):
    """
    Paper-style LSTM + per-horizon Mixture Density Network (MDN) adapted to
    robot-relative planar motion.

    Before calling it, express the observed robot history in the robot frame at
    the current prediction anchor so that:

        current position = (0, 0)
        current heading  = +Z

    Default input features:
        0: relative_x
        1: relative_z
        2: relative_velocity_x
        3: relative_velocity_z
        4: sin(relative_yaw)
        5: cos(relative_yaw)
        6: yaw_rate

    relative_yaw is the historical robot heading relative to the heading at the
    current prediction anchor. Its sin/cos representation avoids angle wrapping
    discontinuities, while yaw_rate explicitly exposes turning dynamics.

    For EACH future timestep independently, the network predicts K correlated
    bivariate Gaussian components over future (x, z):

        mu_x, mu_z, sigma_x, sigma_z, rho, mixture_logit

    Therefore the raw forward output has shape:
        (B, prediction_steps, num_mixtures * 6)

    Use ``parameters_from_output`` to obtain the structured tensors used by the
    loss/evaluation code.
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 8,
        num_layers: int = 1,
        prediction_steps: int = 40,
        num_mixtures: int = 3,
        min_std: float = 1e-3,
        max_log_std: float = 5.0,
        rho_limit: float = 0.999,
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if prediction_steps <= 0:
            raise ValueError("prediction_steps must be positive.")
        if num_mixtures <= 0:
            raise ValueError("num_mixtures must be positive.")
        if min_std <= 0.0:
            raise ValueError("min_std must be positive.")
        if not 0.0 < rho_limit < 1.0:
            raise ValueError("rho_limit must be in (0, 1).")

        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.prediction_steps = int(prediction_steps)
        self.num_mixtures = int(num_mixtures)
        self.min_std = float(min_std)
        self.max_log_std = float(max_log_std)
        self.rho_limit = float(rho_limit)

        self.output_factor = 6
        self.output_size_per_step = self.num_mixtures * self.output_factor

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
        )

        self.fc = nn.Linear(
            self.hidden_size,
            self.prediction_steps * self.output_size_per_step,
        )

        # Match the released implementation's explicit Xavier/zero init.
        for name, parameter in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(parameter)
            elif "bias" in name:
                nn.init.zeros_(parameter)

        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, history_steps, input_size)

        Returns:
            raw MDN output: (B, prediction_steps, num_mixtures * 6)
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, history_steps, {self.input_size}), "
                f"got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got {x.shape[-1]}."
            )

        lstm_out, _ = self.lstm(x)
        encoded = lstm_out[:, -1, :]
        raw = self.fc(encoded)

        return raw.view(
            x.shape[0],
            self.prediction_steps,
            self.output_size_per_step,
        )

    def parameters_from_output(self, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert raw output into per-horizon GMM parameters.

        Returns tensors with shapes:
            mean:          (B, T, K, 2)
            std:           (B, T, K, 2)
            rho:           (B, T, K)
            logits:        (B, T, K)
            probabilities: (B, T, K)
        """
        if output.ndim != 3:
            raise ValueError("MDN output must have shape (B, T, K*6).")

        k = self.num_mixtures

        mu_x = output[..., 0:k]
        mu_z = output[..., k:2 * k]

        log_sigma_x = torch.clamp(
            output[..., 2 * k:3 * k],
            min=-self.max_log_std,
            max=self.max_log_std,
        )
        log_sigma_z = torch.clamp(
            output[..., 3 * k:4 * k],
            min=-self.max_log_std,
            max=self.max_log_std,
        )

        sigma_x = torch.exp(log_sigma_x).clamp_min(self.min_std)
        sigma_z = torch.exp(log_sigma_z).clamp_min(self.min_std)

        rho = self.rho_limit * torch.tanh(
            output[..., 4 * k:5 * k]
        )

        logits = output[..., 5 * k:6 * k]
        probabilities = torch.softmax(logits, dim=-1)

        mean = torch.stack([mu_x, mu_z], dim=-1)
        std = torch.stack([sigma_x, sigma_z], dim=-1)

        return {
            "mean": mean,
            "std": std,
            "rho": rho,
            "logits": logits,
            "probabilities": probabilities,
        }

    def expected_position(self, output: torch.Tensor) -> torch.Tensor:
        """Probability-weighted mean position at each horizon: (B, T, 2)."""
        params = self.parameters_from_output(output)
        return (
            params["probabilities"].unsqueeze(-1) * params["mean"]
        ).sum(dim=2)

    def modal_component_position(self, output: torch.Tensor) -> torch.Tensor:
        """
        Mean of the highest-weight Gaussian at each horizon: (B, T, 2).

        Note: because this is a per-horizon MDN, the selected component index is
        allowed to differ from one future timestep to the next.
        """
        params = self.parameters_from_output(output)
        indices = torch.argmax(params["probabilities"], dim=-1)
        gather_index = indices[..., None, None].expand(-1, -1, 1, 2)

        return torch.gather(
            params["mean"],
            dim=2,
            index=gather_index,
        ).squeeze(2)

    def get_config(self) -> dict:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "prediction_steps": self.prediction_steps,
            "num_mixtures": self.num_mixtures,
            "min_std": self.min_std,
            "max_log_std": self.max_log_std,
            "rho_limit": self.rho_limit,
        }
