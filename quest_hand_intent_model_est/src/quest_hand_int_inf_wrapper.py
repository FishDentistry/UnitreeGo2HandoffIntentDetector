"""Inference and differentiable scoring for the Quest hand-intent MLP.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Union

import numpy as np
import torch

from quest_hand_intent_model_est.src.quest_hand_intent_est_mlp import (
    QuestHandIntentEstimatorMLP,
)
from quest_hand_intent_model_est.src.quest_joint_features import (
    QUEST_JOINT_ORDER,
    extract_quest_joint_features,
    extract_quest_joint_features_json,
)


PathLike = Union[str, Path]
FeatureInput = Union[Sequence[float],Sequence[Sequence[float]],np.ndarray,torch.Tensor]


@dataclass(frozen=True)
class HandIntentPrediction:
    """One hand-intent prediction."""

    probability: float
    is_handoff: bool
    threshold: float

    @property
    def label(self) -> int:
        """Return the binary prediction as 0 or 1."""
        return int(self.is_handoff)

    def as_dict(self) -> dict[str, float | bool | int]:
        """Return a JSON-serializable representation."""
        return {
            "probability": self.probability,
            "is_handoff": self.is_handoff,
            "label": self.label,
            "threshold": self.threshold,
        }


class QuestHandIntentEstInference:
    """Load a trained Quest MLP for inference and differentiable scoring.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pth`` checkpoint written by the training code.
    device:
        ``"auto"`` selects CUDA when available, otherwise CPU. A standard
        PyTorch device string such as ``"cpu"`` or ``"cuda:0"`` is also valid.
    threshold:
        Optional decision-threshold override. When omitted, the threshold saved
        in the checkpoint is used, falling back to 0.5 for older checkpoints.

    Notes
    -----
    The model parameters are frozen because this class is not intended to train
    the model. Freezing the parameters does *not* prevent gradients from being
    computed with respect to input feature tensors during counterfactual
    optimization.
    """

    def __init__(
        self,
        checkpoint_path: PathLike,
        *,
        device: str | torch.device = "auto",
        threshold: float | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.device = self._resolve_device(device)

        checkpoint = self._load_checkpoint(self.checkpoint_path)
        self._validate_checkpoint(checkpoint)
        self.checkpoint: Mapping[str, Any] = checkpoint

        self.input_dim = int(checkpoint["input_dim"])
        self.include_rotations = bool(
            checkpoint.get("include_rotations", False)
        )
        self.features_per_joint = int(
            checkpoint.get(
                "features_per_joint",
                7 if self.include_rotations else 3,
            )
        )
        self.shoulder_centered = bool(
            checkpoint.get("shoulder_centered", True)
        )
        self.joint_order = tuple(
            checkpoint.get("joint_order", QUEST_JOINT_ORDER)
        )

        checkpoint_threshold = float(checkpoint.get("threshold", 0.5))
        self.threshold = self._validate_threshold(
            checkpoint_threshold if threshold is None else threshold
        )

        self._validate_feature_metadata()

        self.model = QuestHandIntentEstimatorMLP(
            input_size=self.input_dim,
            output_size=1,
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Counterfactual optimization needs gradients with respect to the input,
        # not with respect to the trained model parameters.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    # ------------------------------------------------------------------
    # Feature conversion and differentiable model access
    # ------------------------------------------------------------------

    def make_feature_tensor(
        self,
        features: FeatureInput,
        *,
        require_single: bool = False,
    ) -> torch.Tensor:
        """Convert features to a validated float tensor on the model device.

        This method does not intentionally detach an existing tensor, so a
        computational graph leading into ``features`` is preserved.

        Parameters
        ----------
        features:
            Shape ``(D,)`` for one sample or ``(N, D)`` for a batch.
        require_single:
            When true, require exactly one sample.

        Returns
        -------
        torch.Tensor
            Shape ``(N, D)`` on ``self.device``.
        """
        if isinstance(features, torch.Tensor):
            tensor = features.to(device=self.device, dtype=torch.float32)
        else:
            array = np.asarray(features, dtype=np.float32)
            tensor = torch.as_tensor(
                array,
                dtype=torch.float32,
                device=self.device,
            )

        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 2:
            raise ValueError(
                "features must have shape (D,) or (N, D), "
                f"but received {tuple(tensor.shape)}."
            )

        if require_single and int(tensor.shape[0]) != 1:
            raise ValueError(
                "Expected exactly one feature vector, but received "
                f"{int(tensor.shape[0])} samples."
            )

        if int(tensor.shape[1]) != self.input_dim:
            raise ValueError(
                f"Model expects {self.input_dim} features per sample, but "
                f"received {int(tensor.shape[1])}. The checkpoint was trained "
                f"with include_rotations={self.include_rotations}."
            )

        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("features contain NaN or infinite values.")

        return tensor

    def make_optimizable_feature_tensor(
        self,
        features: FeatureInput,
        *,
        require_single: bool = True,
    ) -> torch.Tensor:
        """Create a leaf feature tensor that can be passed to an optimizer.

        This is useful when the feature vector itself is the optimization
        variable. The returned tensor is detached from the original input,
        cloned, and configured with ``requires_grad=True``.
        """
        tensor = self.make_feature_tensor(
            features,
            require_single=require_single,
        )
        return tensor.detach().clone().requires_grad_(True)

    def forward_feature_probabilities(
        self,
        feature_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Return differentiable handoff probabilities for feature tensors.

        Unlike the ``predict_*`` methods, this method does not use
        ``torch.inference_mode()``, does not detach the result, and does not
        convert it to NumPy. Gradients can therefore flow from the returned
        probabilities back to ``feature_tensor``.

        Parameters
        ----------
        feature_tensor:
            Tensor with shape ``(D,)`` or ``(N, D)``.

        Returns
        -------
        torch.Tensor
            One-dimensional tensor of shape ``(N,)``.
        """
        features = self.make_feature_tensor(feature_tensor)
        probabilities = self.model(features).reshape(-1)

        if not bool(torch.isfinite(probabilities).all()):
            raise RuntimeError(
                "The model produced NaN or infinite probabilities."
            )

        # The training code uses BCELoss and expects a sigmoid probability.
        # Avoid converting to Python/NumPy here because that would break the
        # gradient graph.
        return probabilities

    # ------------------------------------------------------------------
    # Quick prediction from already-extracted feature vectors
    # ------------------------------------------------------------------

    def predict_feature_probabilities(
        self,
        features: FeatureInput,
    ) -> np.ndarray:
        """Predict probabilities from one feature vector or a feature batch.

        This is a no-gradient convenience method. Use
        ``forward_feature_probabilities`` during counterfactual optimization.
        """
        feature_tensor = self.make_feature_tensor(features)

        with torch.inference_mode():
            probabilities = self.model(feature_tensor).reshape(-1)

        probabilities_array = (
            probabilities.detach().cpu().numpy().astype(np.float32)
        )
        self._validate_probability_array(probabilities_array)
        return probabilities_array

    def predict_features(
        self,
        features: FeatureInput,
        *,
        threshold: float | None = None,
    ) -> HandIntentPrediction:
        """Predict from one already-extracted feature vector."""
        probabilities = self.predict_feature_probabilities(features)
        if len(probabilities) != 1:
            raise ValueError(
                "predict_features expects exactly one sample. Use "
                "predict_feature_batch for multiple samples."
            )
        return self._make_prediction(probabilities[0], threshold)

    def predict_feature_batch(
        self,
        feature_batch: FeatureInput,
        *,
        threshold: float | None = None,
    ) -> list[HandIntentPrediction]:
        """Predict from a batch of already-extracted feature vectors."""
        probabilities = self.predict_feature_probabilities(feature_batch)
        return [
            self._make_prediction(value, threshold)
            for value in probabilities
        ]

    # ------------------------------------------------------------------
    # Feature extraction and prediction from Quest joint data
    # ------------------------------------------------------------------

    def features_from_joint_json(
        self,
        joint_payload: Mapping[str, Any],
    ) -> np.ndarray:
        """Extract features from an in-memory Quest joint JSON object/dict."""
        extractor_parameters = inspect.signature(
            extract_quest_joint_features_json
        ).parameters
        if "include_rotations" in extractor_parameters:
            feature_vector = extract_quest_joint_features_json(
                joint_payload,
                include_rotations=self.include_rotations,
            )
        else:
            # Compatibility with an older extractor whose JSON variant did not
            # expose the include_rotations keyword. Its output is still checked
            # against the checkpoint input dimension below.
            feature_vector = extract_quest_joint_features_json(joint_payload)
        return self._validate_extracted_feature_vector(
            feature_vector,
            source_description="in-memory joint JSON",
        )

    def features_from_joint_json_file(
        self,
        joint_json_path: PathLike,
    ) -> np.ndarray:
        """Extract features from a Quest joint JSON file path."""
        feature_vector = extract_quest_joint_features(
            joint_json_path,
            include_rotations=self.include_rotations,
        )
        return self._validate_extracted_feature_vector(
            feature_vector,
            source_description=str(joint_json_path),
        )

    def predict_joint_json(
        self,
        joint_payload: Mapping[str, Any],
        *,
        threshold: float | None = None,
    ) -> HandIntentPrediction:
        """Extract features from an in-memory joint JSON object and predict."""
        features = self.features_from_joint_json(joint_payload)
        return self.predict_features(features, threshold=threshold)

    def predict_joint_json_file(
        self,
        joint_json_path: PathLike,
        *,
        threshold: float | None = None,
    ) -> HandIntentPrediction:
        """Extract features from one joint JSON file path and predict."""
        features = self.features_from_joint_json_file(joint_json_path)
        return self.predict_features(features, threshold=threshold)

    def predict_joint_json_files(
        self,
        joint_json_paths: Iterable[PathLike],
        *,
        threshold: float | None = None,
        batch_size: int = 256,
    ) -> list[HandIntentPrediction]:
        """Predict multiple joint JSON file paths in mini-batches."""
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        paths = list(joint_json_paths)
        if not paths:
            return []

        feature_matrix = np.stack(
            [self.features_from_joint_json_file(path) for path in paths]
        ).astype(np.float32)

        results: list[HandIntentPrediction] = []
        for start in range(0, len(feature_matrix), int(batch_size)):
            batch = feature_matrix[start : start + int(batch_size)]
            results.extend(
                self.predict_feature_batch(batch, threshold=threshold)
            )
        return results

    # ------------------------------------------------------------------
    # Backward-compatible aliases
    # ------------------------------------------------------------------

    def extract_features(self, joint_json_path: PathLike) -> np.ndarray:
        """Backward-compatible alias for ``features_from_joint_json_file``."""
        return self.features_from_joint_json_file(joint_json_path)

    def predict_joint_file(
        self,
        joint_json_path: PathLike,
        *,
        threshold: float | None = None,
    ) -> HandIntentPrediction:
        """Backward-compatible alias for ``predict_joint_json_file``."""
        return self.predict_joint_json_file(
            joint_json_path,
            threshold=threshold,
        )

    def predict_joint_files(
        self,
        joint_json_paths: Iterable[PathLike],
        *,
        threshold: float | None = None,
        batch_size: int = 256,
    ) -> list[HandIntentPrediction]:
        """Backward-compatible alias for ``predict_joint_json_files``."""
        return self.predict_joint_json_files(
            joint_json_paths,
            threshold=threshold,
            batch_size=batch_size,
        )

    def predict_proba(self, joint_json_path: PathLike) -> float:
        """Return only the probability for a joint JSON file path.

        Kept for backward compatibility. The more explicit method is
        ``predict_joint_json_file(path).probability``.
        """
        return self.predict_joint_json_file(joint_json_path).probability

    def predict(
        self,
        joint_json_path: PathLike,
        *,
        threshold: float | None = None,
    ) -> bool:
        """Return only the decision for a joint JSON file path.

        Kept for backward compatibility. The more explicit method is
        ``predict_joint_json_file(path).is_handoff``.
        """
        return self.predict_joint_json_file(
            joint_json_path,
            threshold=threshold,
        ).is_handoff

    def __call__(
        self,
        joint_json_path: PathLike,
        *,
        threshold: float | None = None,
    ) -> HandIntentPrediction:
        """Backward-compatible alias for ``predict_joint_json_file``."""
        return self.predict_joint_json_file(
            joint_json_path,
            threshold=threshold,
        )

    # ------------------------------------------------------------------
    # Validation and metadata
    # ------------------------------------------------------------------

    def _validate_extracted_feature_vector(
        self,
        feature_vector: Sequence[float] | np.ndarray | torch.Tensor,
        *,
        source_description: str,
    ) -> np.ndarray:
        array = np.asarray(feature_vector, dtype=np.float32)
        if array.ndim != 1:
            raise ValueError(
                "Feature extraction must return a 1D vector, but returned "
                f"shape {array.shape} for {source_description}."
            )

        self.make_feature_tensor(array, require_single=True)
        return array

    @staticmethod
    def _validate_probability_array(probabilities: np.ndarray) -> None:
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError(
                "The model produced NaN or infinite probabilities."
            )

        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise RuntimeError(
                "The model output falls outside [0, 1]. The wrapper expects "
                "QuestHandIntentEstimatorMLP to return sigmoid probabilities."
            )

    def _make_prediction(
        self,
        probability: float | np.floating,
        threshold: float | None,
    ) -> HandIntentPrediction:
        resolved_threshold = (
            self.threshold
            if threshold is None
            else self._validate_threshold(threshold)
        )
        probability_float = float(probability)
        return HandIntentPrediction(
            probability=probability_float,
            is_handoff=probability_float >= resolved_threshold,
            threshold=resolved_threshold,
        )

    @staticmethod
    def _resolve_device(device: str | torch.device) -> torch.device:
        if isinstance(device, torch.device):
            resolved = device
        elif device == "auto":
            resolved = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            resolved = torch.device(device)

        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {resolved} was requested, but CUDA is unavailable."
            )

        return resolved

    @staticmethod
    def _load_checkpoint(path: Path) -> Mapping[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")

        try:
            checkpoint = torch.load(
                path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")

        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                "Expected the checkpoint to be a mapping, but received "
                f"{type(checkpoint).__name__}."
            )

        return checkpoint

    @staticmethod
    def _validate_checkpoint(checkpoint: Mapping[str, Any]) -> None:
        required_keys = {"input_dim", "model_state_dict"}
        missing = sorted(required_keys.difference(checkpoint))
        if missing:
            raise KeyError(
                "Checkpoint is missing required key(s): " + ", ".join(missing)
            )

    @staticmethod
    def _validate_threshold(value: float) -> float:
        threshold = float(value)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be finite and in [0, 1], got {value!r}."
            )
        return threshold

    def _validate_feature_metadata(self) -> None:
        expected_dim = len(self.joint_order) * self.features_per_joint
        if expected_dim != self.input_dim:
            raise ValueError(
                "Checkpoint feature metadata is inconsistent: "
                f"len(joint_order)={len(self.joint_order)}, "
                f"features_per_joint={self.features_per_joint}, so the expected "
                f"input dimension is {expected_dim}, but input_dim={self.input_dim}."
            )

        runtime_joint_order = tuple(QUEST_JOINT_ORDER)
        if self.joint_order != runtime_joint_order:
            raise ValueError(
                "The checkpoint joint order does not match the currently imported "
                "QUEST_JOINT_ORDER. Use the same feature-extraction code/version "
                "that was used during training.\n"
                f"Checkpoint: {self.joint_order}\n"
                f"Runtime:    {runtime_joint_order}"
            )

    @property
    def metadata(self) -> dict[str, Any]:
        """Return checkpoint metadata relevant to inference."""
        return {
            "checkpoint_path": str(self.checkpoint_path),
            "device": str(self.device),
            "input_dim": self.input_dim,
            "joint_order": list(self.joint_order),
            "shoulder_centered": self.shoulder_centered,
            "include_rotations": self.include_rotations,
            "features_per_joint": self.features_per_joint,
            "threshold": self.threshold,
            "target_type": self.checkpoint.get("target_type"),
        }