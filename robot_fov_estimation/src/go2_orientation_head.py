import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Go2OrientationHead(nn.Module):
    """
    Predict the robot's view-relative horizontal orientation from a ResNet
    embedding.

    Input:
        embeddings: Tensor of shape (B, embedding_dim)

    Output:
        Tensor of shape (B, 2), normalized to unit length:

            output[:, 0] = sin(alpha)
            output[:, 1] = cos(alpha)

    where alpha is the signed horizontal angle from the camera->robot
    direction to the robot's forward direction.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.embedding_dim = int(
            embedding_dim
        )

        self.hidden_dim = int(
            hidden_dim
        )

        self.dropout = float(
            dropout
        )

        self.net = nn.Sequential(
            nn.Linear(
                self.embedding_dim,
                self.hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(
                self.dropout
            ),
            nn.Linear(
                self.hidden_dim,
                2,
            ),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(
                "Expected embeddings with shape "
                f"(B, {self.embedding_dim}), "
                f"but received {tuple(embeddings.shape)}."
            )

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.embedding_dim}, "
                f"received {embeddings.shape[1]}."
            )

        output = self.net(
            embeddings
        )

        # Interpret the two outputs as a 2D unit vector:
        #
        #     [sin(alpha), cos(alpha)]
        #
        # Normalization makes the representation explicitly circular and
        # prevents the network from changing the output magnitude instead of
        # learning the angle.
        output = F.normalize(
            output,
            p=2,
            dim=-1,
            eps=1.0e-8,
        )

        return output


def orientation_cosine_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Circular angular loss for normalized [sin(alpha), cos(alpha)] targets.

    For angular error delta:

        loss = 1 - cos(delta)

    This naturally handles wraparound at -180 / +180 degrees.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            "Prediction and target shapes must match: "
            f"{tuple(prediction.shape)} vs "
            f"{tuple(target.shape)}."
        )

    prediction = F.normalize(
        prediction,
        p=2,
        dim=-1,
        eps=1.0e-8,
    )

    target = F.normalize(
        target,
        p=2,
        dim=-1,
        eps=1.0e-8,
    )

    cosine_similarity = (
        prediction
        * target
    ).sum(
        dim=-1
    )

    return (
        1.0
        - cosine_similarity
    ).mean()


def sincos_to_angle_radians(
    prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Convert (..., 2) [sin(alpha), cos(alpha)] predictions to radians
    in [-pi, pi].
    """
    return torch.atan2(
        prediction[..., 0],
        prediction[..., 1],
    )


def sincos_to_angle_degrees(
    prediction: torch.Tensor,
) -> torch.Tensor:
    """
    Convert (..., 2) [sin(alpha), cos(alpha)] predictions to degrees
    in [-180, 180].
    """
    radians = (
        sincos_to_angle_radians(
            prediction
        )
    )

    return torch.rad2deg(
        radians
    )


def angular_error_degrees(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Return the absolute wrapped angular error in degrees for each sample.

    prediction and target are both (..., 2) tensors encoded as:

        [sin(alpha), cos(alpha)]
    """
    prediction_angle = (
        sincos_to_angle_radians(
            prediction
        )
    )

    target_angle = (
        sincos_to_angle_radians(
            target
        )
    )

    delta = (
        prediction_angle
        - target_angle
    )

    wrapped_delta = torch.atan2(
        torch.sin(
            delta
        ),
        torch.cos(
            delta
        ),
    )

    return torch.abs(
        torch.rad2deg(
            wrapped_delta
        )
    )


if __name__ == "__main__":
    # Small shape/functionality test.
    batch_size = 8

    embeddings = torch.randn(
        batch_size,
        512,
    )

    model = Go2OrientationHead()

    predictions = model(
        embeddings
    )

    print(
        "Prediction shape:",
        predictions.shape,
    )

    print(
        "Prediction norms:",
        torch.linalg.vector_norm(
            predictions,
            dim=-1,
        ),
    )

    print(
        "Predicted angles (deg):",
        sincos_to_angle_degrees(
            predictions
        ),
    )