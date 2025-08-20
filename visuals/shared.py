from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SharedBands:
	"""Thread-safe container for the latest band magnitudes in [0,1]."""
	values: Optional[np.ndarray] = None
	lock: threading.Lock = threading.Lock()

	def set(self, arr: np.ndarray) -> None:
		with self.lock:
			self.values = np.asarray(arr, dtype=np.float32).copy()

	def get(self) -> Optional[np.ndarray]:
		with self.lock:
			if self.values is None:
				return None
			return self.values.copy()


