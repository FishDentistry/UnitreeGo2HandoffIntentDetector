from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForestClassifier


ArrayLike = Union[np.ndarray, Iterable[Iterable[float]]]
PathLike = Union[str, Path]


@dataclass
class RandomForestConfig:
	"""Configuration for a standard random forest classifier."""

	n_estimators: int = 200
	max_depth: Optional[int] = None
	min_samples_split: int = 2
	min_samples_leaf: int = 1
	max_features: Union[str, int, float, None] = "sqrt"
	class_weight: Optional[Union[Dict[int, float], str]] = "balanced"
	random_state: Optional[int] = 0
	n_jobs: Optional[int] = -1


class RandomForestClassifier:
	"""Train and run inference with a standard random forest model.

	This wraps a `RandomForestClassifier` so training and inference are
	consistent and easy to reuse.
	"""

	def __init__(self, config: Optional[RandomForestConfig] = None) -> None:
		self.config = config or RandomForestConfig()
		self.model = SklearnRandomForestClassifier(
			n_estimators=self.config.n_estimators,
			max_depth=self.config.max_depth,
			min_samples_split=self.config.min_samples_split,
			min_samples_leaf=self.config.min_samples_leaf,
			max_features=self.config.max_features,
			class_weight=self.config.class_weight,
			random_state=self.config.random_state,
			n_jobs=self.config.n_jobs,
		)
		self._is_fitted = False

	def fit(self, x: ArrayLike, y: Union[np.ndarray, Iterable[int]]) -> "RandomForestClassifier":
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

	def predict_proba(self, x: ArrayLike) -> np.ndarray:
		self._require_fitted()
		x_np = np.asarray(x, dtype=np.float32)
		return self.model.predict_proba(x_np)

	def infer(self, x: ArrayLike) -> Dict[str, np.ndarray]:
		"""Convenience inference API returning labels and score values."""
		predictions = self.predict(x)
		probabilities = self.predict_proba(x)

		if probabilities.ndim == 2 and probabilities.shape[1] > 1:
			scores = probabilities[:, 1]
		else:
			scores = np.asarray(probabilities)

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
	def load(cls, path: PathLike) -> "RandomForestClassifier":
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
