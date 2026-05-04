import os
import json
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

# =====================
# PATHS
# =====================
TRAJ_DIR = "trajectories"
OUTPUT_JSON = "stop_lines.json"

# =====================
# PARAMETERS (robust defaults for DLR)
# =====================
STOP_SPEED_THRESH = 0.2        # m/s
MIN_STOP_DURATION = 5.0        # seconds
MAX_YAW_STD_DEG = 5.0          # degrees
MAX_STOP_DISPLACEMENT = 0.5    # meters

DBSCAN_EPS = 1.0               # meters
DBSCAN_MIN_SAMPLES = 30

# =====================
# COLUMN NAMES
# =====================
COL_TIME  = "timestamp"
COL_ID    = "id"
COL_X     = "center_easting"
COL_Y     = "center_northing"
COL_YAW   = "yaw"
COL_SPEED = "velocity_magnitude"   # optional

# =====================
# UTILITIES
# =====================

def timestamps_to_seconds(ts: pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts)
    return (ts - ts.iloc[0]).dt.total_seconds().values


def ensure_speed(df: pd.DataFrame) -> pd.DataFrame:
    if COL_SPEED in df.columns:
        return df

    df = df.sort_values(COL_TIME).copy()
    t = timestamps_to_seconds(df[COL_TIME])

    dx = df[COL_X].diff().fillna(0.0).values
    dy = df[COL_Y].diff().fillna(0.0).values
    dt = np.diff(t, prepend=t[1])

    speed = np.hypot(dx, dy) / np.maximum(dt, 1e-3)
    df[COL_SPEED] = speed
    return df


def detect_stop_events(agent_df: pd.DataFrame):
    """
    Detect true traffic-light stop events using speed + yaw stability.
    Returns list of (x, y, t_sec).
    """
    stops = []
    buffer = []

    agent_df = agent_df.sort_values(COL_TIME).copy()
    t_sec = timestamps_to_seconds(agent_df[COL_TIME])
    agent_df["t_sec"] = t_sec

    for _, row in agent_df.iterrows():
        if row[COL_SPEED] < STOP_SPEED_THRESH:
            buffer.append(row)
        else:
            stops.extend(_flush_buffer(buffer))
            buffer = []

    stops.extend(_flush_buffer(buffer))
    return stops


def _flush_buffer(buffer):
    if len(buffer) < 2:
        return []

    duration = buffer[-1]["t_sec"] - buffer[0]["t_sec"]
    if duration < MIN_STOP_DURATION:
        return []

    # spatial stability
    x0, y0 = buffer[0][COL_X], buffer[0][COL_Y]
    x1, y1 = buffer[-1][COL_X], buffer[-1][COL_Y]
    if np.hypot(x1 - x0, y1 - y0) > MAX_STOP_DISPLACEMENT:
        return []

    # yaw stability
    yaws = np.unwrap(np.deg2rad([r[COL_YAW] for r in buffer]))
    yaw_std = np.rad2deg(np.std(yaws))
    if yaw_std > MAX_YAW_STD_DEG:
        return []

    mid = buffer[len(buffer)//2]
    return [(mid[COL_X], mid[COL_Y], mid["t_sec"])]


def compute_stop_line_geometry(points_xy: np.ndarray):
    center = points_xy.mean(axis=0)

    pca = PCA(n_components=2)
    pca.fit(points_xy)
    direction = pca.components_[0]

    yaw = float(np.arctan2(direction[1], direction[0]))
    return float(center[0]), float(center[1]), yaw


# =====================
# MAIN PIPELINE
# =====================

def extract_stop_lines():
    csv_files = sorted(glob(os.path.join(TRAJ_DIR, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No trajectory CSVs found in {TRAJ_DIR}")

    print(f"[INFO] Found {len(csv_files)} trajectory CSV files")

    all_stop_events = []

    for csv_path in tqdm(csv_files, desc="Processing trajectories"):
        df = pd.read_csv(csv_path)

        df = ensure_speed(df)

        for _, agent_df in df.groupby(COL_ID):
            stops = detect_stop_events(agent_df)
            all_stop_events.extend(stops)

    if not all_stop_events:
        raise RuntimeError("No valid traffic-light stop events detected.")

    stop_xy = np.array([(x, y) for x, y, _ in all_stop_events])

    print(f"[INFO] Clustering {len(stop_xy)} stop points")

    clustering = DBSCAN(
        eps=DBSCAN_EPS,
        min_samples=DBSCAN_MIN_SAMPLES
    ).fit(stop_xy)

    labels = clustering.labels_
    cluster_ids = sorted(cid for cid in set(labels) if cid != -1)

    print(f"[INFO] Found {len(cluster_ids)} stop-line clusters")

    stop_lines = []

    for cid in cluster_ids:
        pts = stop_xy[labels == cid]
        cx, cy, yaw = compute_stop_line_geometry(pts)

        stop_lines.append({
            "id": int(cid),
            "x": cx,
            "y": cy,
            "yaw": yaw,
            "num_points": int(len(pts))
        })

    with open(OUTPUT_JSON, "w") as f:
        json.dump({"stop_lines": stop_lines}, f, indent=2)

    print(f"\n✅ Stop-line geometry written to {OUTPUT_JSON}")


if __name__ == "__main__":
    extract_stop_lines()
