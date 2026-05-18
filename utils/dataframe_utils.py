# utils/dataframe_utils.py
from pathlib import Path
import json
import pandas as pd
from utils.timestamp_utils import extract_ts_token
from config import CSV_FILE_PATTERN, BOUNDS_FILE
from core.frame import Frame
from typing import Generator
from tqdm import tqdm

# # -------- lazy frame generator --------------------------------------------
# def iter_frames(csv_root: Path) -> Generator[Frame, None, None]:
#     """
#     Stream Frame objects chronologically *file-by-file*.
#     Assumes each CSV is already sorted by timestamp (true for the
#     original dataset).  If needed, you can sort each group explicitly.
#     """
#     csv_paths = sorted(csv_root.glob(CSV_FILE_PATTERN))
#     if not csv_paths:
#         raise FileNotFoundError(f"No CSVs in {csv_root} matching {CSV_FILE_PATTERN}")

#     for csv_path in csv_paths:
#         df_file = pd.read_csv(csv_path, parse_dates=["timestamp"])
#         for ts, grp in df_file.groupby("timestamp", sort=True):
#             yield Frame.from_df(ts, grp)


# -------- lazy frame generator --------------------------------------------
def iter_frames(csv_root: Path) -> Generator[Frame, None, None]:
    """
    Stream Frame objects chronologically *file-by-file*.
    Assumes each CSV is already sorted by timestamp (true for the
    original dataset).  If needed, you can sort each group explicitly.
    """
    csv_paths = sorted(csv_root.glob(CSV_FILE_PATTERN))
    if not csv_paths:
        raise FileNotFoundError(f"No CSVs in {csv_root} matching {CSV_FILE_PATTERN}")

    for csv_path in tqdm(csv_paths, desc="CSV files"):
        df_file = pd.read_csv(csv_path, parse_dates=["timestamp"])

        # group progress inside each file
        groups = list(df_file.groupby("timestamp", sort=True))

        for ts, grp in tqdm(groups, desc=f"Frames ({csv_path.name})", leave=False):
            yield Frame.from_df(ts, grp)

def load_dataframe(csv_root: Path) -> pd.DataFrame:
    paths = sorted(csv_root.glob(CSV_FILE_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No CSVs in {csv_root} matching {CSV_FILE_PATTERN}")
    df = pd.concat([pd.read_csv(p, parse_dates=["timestamp"]) for p in paths],
                   ignore_index=True)
    return df

# --- bounds helpers --------------------------------------------------------
def compute_bounds(df: pd.DataFrame, buffer: float = 0.05) -> dict:
    mn_e, mx_e = df["center_easting"].min(), df["center_easting"].max()
    mn_n, mx_n = df["center_northing"].min(), df["center_northing"].max()
    buf_e = (mx_e - mn_e) * buffer or 1.0
    buf_n = (mx_n - mn_n) * buffer or 1.0
    return {"left": mn_e - buf_e, "right": mx_e + buf_e,
            "bottom": mn_n - buf_n, "top": mx_n + buf_n}

def load_or_compute_bounds(df: pd.DataFrame, path: Path = Path(BOUNDS_FILE)) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    bounds = compute_bounds(df)
    path.write_text(json.dumps(bounds, indent=2))
    return bounds
# ---------------------------------------------------------------------------
def load_or_compute_bounds_hdf(
    h5_path: Path,
    json_path: Path = Path("global_bounds.json"),
    buffer: float = 0.05,
) -> dict:
    """
    Return global bounds from pre-saved JSON if present; otherwise compute
    across the whole HDF table (streaming one column at a time, so memory ≪ dataset).
    """
    import json, pandas as pd
    if json_path.exists():
        return json.loads(json_path.read_text())

    with pd.HDFStore(h5_path, mode="r") as store:
        # read only the needed numeric columns as Series (still fast)
        ce = store.select_column("traj", "center_easting")
        cn = store.select_column("traj", "center_northing")

    mn_e, mx_e = ce.min(), ce.max()
    mn_n, mx_n = cn.min(), cn.max()
    buf_e = (mx_e - mn_e) * buffer or 1.0
    buf_n = (mx_n - mn_n) * buffer or 1.0

    bounds = {"left": mn_e - buf_e, "right": mx_e + buf_e,
              "bottom": mn_n - buf_n, "top": mx_n + buf_n}
    json_path.write_text(json.dumps(bounds, indent=2))
    return bounds


def frames_from_df(df: pd.DataFrame) -> list[Frame]:
    """Group the big trajectory table into Frame objects (timestamp order)."""
    frames: list[Frame] = []
    for ts, grp in df.groupby("timestamp", sort=True):
        frames.append(Frame.from_df(ts, grp))
    return frames