from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ArrayLike = Union[np.ndarray, Iterable[Iterable[float]]]
PathLike = Union[str, Path]


@dataclass
class LinearSVMConfig:
    """Configuration for a standard linear SVM classifier."""

    c: float = 1.0
    class_weight: Optional[Union[Dict[int, float], str]] = "balanced"
    random_state: Optional[int] = 0
    max_iter: int = 20000


class LinearSVMClassifier:
    """Train and run inference with a standard linear SVM model.

    This wraps a `StandardScaler -> LinearSVC` pipeline so training and
    inference are consistent and easy to reuse.
    """

    def __init__(self, config: Optional[LinearSVMConfig] = None) -> None:
        self.config = config or LinearSVMConfig()
        self.model: Pipeline = make_pipeline(
            StandardScaler(),
            LinearSVC(
                C=self.config.c,
                class_weight=self.config.class_weight,
                random_state=self.config.random_state,
                max_iter=self.config.max_iter,
            ),
        )
        self._is_fitted = False

    def fit(self, x: ArrayLike, y: Union[np.ndarray, Iterable[int]]) -> "LinearSVMClassifier":
        x_np = np.asarray(x, dtype=np.float32)
        y_np = np.asarray(y)

        if x_np.ndim != 2:
            raise ValueError(f"Expected x to be 2D, got shape {x_np.shape}")
        if y_np.ndim != 1:
            raise ValueError(f"Expected y to be 1D, got shape {y_np.shape}")
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError(
                "Number of rows in x must match number of labels in y "
                f"({x_np.shape[0]} != {y_np.shape[0]})"
            )

        self.model.fit(x_np, y_np)
        self._is_fitted = True
        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        self._require_fitted()
        x_np = np.asarray(x, dtype=np.float32)
        return self.model.predict(x_np)

    def decision_function(self, x: ArrayLike) -> np.ndarray:
        self._require_fitted()
        x_np = np.asarray(x, dtype=np.float32)
        return self.model.decision_function(x_np)

    def infer(self, x: ArrayLike) -> Dict[str, np.ndarray]:
        """Convenience inference API returning labels and decision scores."""
        predictions = self.predict(x)
        scores = self.decision_function(x)
        return {"predictions": predictions, "scores": np.asarray(scores)}

    def save(self, path: PathLike) -> None:
        self._require_fitted()
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with path_obj.open("wb") as f:
            pickle.dump(
                {
                    "config": self.config,
                    "model": self.model,
                    "is_fitted": self._is_fitted,
                },
                f,
            )

    @classmethod
    def load(cls, path: PathLike) -> "LinearSVMClassifier":
        path_obj = Path(path)
        with path_obj.open("rb") as f:
            payload: Dict[str, Any] = pickle.load(f)

        instance = cls(config=payload["config"])
        instance.model = payload["model"]
        instance._is_fitted = payload.get("is_fitted", True)
        return instance

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit(...) before inference.")