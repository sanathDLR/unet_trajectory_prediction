from __future__ import annotations

"""core.scene

A lightweight, backend‑agnostic representation of one RGB composite (multi‑step
image) and its temporal window.

This *core* object **does not** depend on pandas.  It can therefore be shared by
training, evaluation, visualisation, or even a real‑time ROS node without
pulling in heavyweight tabular dependencies.

-------------------------------------------------------------------------
API
-------------------------------------------------------------------------
Scene(
    t0: pd.Timestamp,
    rgb_path: Path,
    dt: float = DT_FRAME,
    steps: int = 3,
)

Attributes
----------
• t0        : base timestamp of the composite (channel 0)
• dt        : seconds between channels
• steps     : number of channels (3 ⇒ t0,t1,t2)
• rgb_path  : file on disk (PNG) holding the composite

Convenience
-----------
• times       → list[pd.Timestamp]              (length = steps)
• scene_id    → str  (YYYYmmdd_HHMMSSmmm token)
• load_rgb()  → torch.Tensor  (C×H×W, float64 in [0,1])

Factory helpers
---------------
• from_rgb_filename(path: Path)      – parse timestamp token
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import pandas as pd
import torch
import cv2
from utils.timestamp_utils import extract_ts_token, parse_ts_token, format_ts_token
from config import DT_FRAME

__all__ = ["Scene"]


@dataclass
class Scene:
    t0: pd.Timestamp
    rgb_path: Path
    dt: float = DT_FRAME
    steps: int = 3  # channels in composite

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------
    @property
    def times(self) -> List[pd.Timestamp]:
        """List of timestamps for each channel (t0 .. t{steps-1})."""
        return [self.t0 + pd.Timedelta(seconds=i * self.dt) for i in range(self.steps)]

    @property
    def scene_id(self) -> str:
        """Filename‑friendly token (YYYYmmdd_HHMMSSmmm)."""
        return format_ts_token(self.t0)
    
    @property
    def path(self) -> Path:
        """Alias for image_path (kept for legacy helpers)."""
        return self.rgb_path

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------
    def load_rgb(self, device: torch.device | None = None) -> torch.Tensor:
        """Read the RGB composite as C×H×W float64 tensor in [0,1]."""
        img_bgr = cv2.imread(str(self.rgb_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(self.rgb_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ten = torch.from_numpy(img_rgb.astype("float64") / 255.0).permute(2, 0, 1)
        return ten if device is None else ten.to(device)

    # ------------------------------------------------------------------
    # Alternate constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_rgb_filename(cls, path: Path, dt: float = DT_FRAME, steps: int = 3) -> Scene:
        token = extract_ts_token(path.name)
        return cls(parse_ts_token(token), path, dt, steps)

    # ------------------------------------------------------------------
    def __repr__(self):  # pragma: no cover
        return f"Scene(t0={self.t0}, steps={self.steps}, path={self.rgb_path.name})"
