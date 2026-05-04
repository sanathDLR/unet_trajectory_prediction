from __future__ import annotations
from pathlib import Path
import pandas as pd
from core.frame import Frame

class HDFFrameProvider:
    """
    Lazy, cached access to per-timestamp Frames from an HDF5 store.
    """
    def __init__(self, h5_path: Path, key: str = "traj"):
        self._store = pd.HDFStore(h5_path, mode="r")
        self._key   = key
        self._cache: dict[pd.Timestamp, Frame] = {}

    def get(self, ts: pd.Timestamp) -> Frame | None:
        if ts in self._cache:
            return self._cache[ts]

        q = f'timestamp == "{ts.isoformat()}"'     # ---- FIXED ----
        df = self._store.select(self._key, where=q)
        if df.empty:
            return None
        fr = Frame.from_df(ts, df)
        self._cache[ts] = fr
        return fr


    def close(self):
        self._store.close()
