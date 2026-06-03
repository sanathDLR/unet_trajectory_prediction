from pathlib import Path
import cv2
import math
import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import glob

from utils.dataframe_utils import iter_frames, load_or_compute_bounds
from core.frame import Frame

# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 256

# None = generate everything
MAX_FRAMES = None

DT_SECONDS = 0.2
ORIGINAL_DT = 0.05
FRAME_SKIP = int(DT_SECONDS / ORIGINAL_DT)

OUT_DIR = Path("dataset_png/frames")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAJ_ROOT = Path(
    "trajectories/DLR-Urban-Traffic-dataset/"
    "DLR-Urban-Traffic-dataset_v1-2-1/raw_data/trajectories/"
)

TL_DIR = Path(
    "trajectories/DLR-Urban-Traffic-dataset/"
    "DLR-Urban-Traffic-dataset_v1-2-1/raw_data/traffic_lights/"
)

STOP_LINE_JSON = "stop_lines.json"
TOPOLOGY_PNG = "lane_topology.png"

STOP_STATES = {3, 4}

# ============================================================
# LOAD BOUNDS
# ============================================================

first_csv = sorted(
    TRAJ_ROOT.glob("trajectories_*.csv")
)[0]

bounds_df = pd.read_csv(
    first_csv,
    nrows=1000
)

bounds = load_or_compute_bounds(bounds_df)

left = bounds["left"]
right = bounds["right"]
bottom = bounds["bottom"]
top = bounds["top"]

scale_x = IMAGE_SIZE / (right - left)
scale_y = IMAGE_SIZE / (top - bottom)

def world_to_pixel(e, n):

    px = int((e - left) * scale_x)
    py = int((top - n) * scale_y)

    px = np.clip(px, 0, IMAGE_SIZE - 1)
    py = np.clip(py, 0, IMAGE_SIZE - 1)

    return px, py

# ============================================================
# STOP LINES
# ============================================================

with open(STOP_LINE_JSON, "r") as f:
    STOP_LINES = json.load(f)["stop_lines"]

def load_traffic_lights():

    csvs = glob.glob(
        str(TL_DIR / "**/*.csv"),
        recursive=True
    )

    dfs = [pd.read_csv(c) for c in csvs]

    df = pd.concat(
        dfs,
        ignore_index=True
    )

    df["timestamp"] = (
        pd.to_datetime(
            df["timestamp"],
            format="mixed",
            utc=True
        ).astype("int64") / 1e9
    )

    tl_states = {}

    for tl_id, g in df.groupby("id"):

        times = g["timestamp"].to_numpy()
        states = g["state"].to_numpy()

        order = np.argsort(times)

        tl_states[int(tl_id)] = (
            times[order],
            states[order]
        )

    return tl_states

TL_STATES = load_traffic_lights()

def is_stop_required(sl_id, timestamp):

    if sl_id not in TL_STATES:
        return False

    times, states = TL_STATES[sl_id]

    idx = np.searchsorted(
        times,
        timestamp,
        side="right"
    ) - 1

    if idx < 0:
        return False

    return states[idx] in STOP_STATES

# ============================================================
# BOX RASTERIZATION
# ============================================================

def oriented_box_corners(
    x,
    y,
    yaw_deg,
    length,
    width
):

    yaw = math.radians(yaw_deg)

    dx = length / 2
    dy = width / 2

    corners = np.array([
        [ dx,  dy],
        [ dx, -dy],
        [-dx, -dy],
        [-dx,  dy],
    ])

    R = np.array([
        [math.cos(yaw), -math.sin(yaw)],
        [math.sin(yaw),  math.cos(yaw)],
    ])

    return corners @ R.T + np.array([x, y])

def rasterize_box(frame: Frame):

    img = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    for ag in frame.agents:

        corners = oriented_box_corners(
            ag.easting,
            ag.northing,
            ag.yaw_deg,
            ag.length_m,
            ag.width_m
        )

        pts = np.array(
            [world_to_pixel(x, y)
             for x, y in corners],
            np.int32
        )

        cv2.fillPoly(
            img,
            [pts],
            255
        )

    return img

# ============================================================
# STOPLINES
# ============================================================

def rasterize_stoplines(timestamp):

    img = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    for sl in STOP_LINES:

        if is_stop_required(
            sl["id"],
            timestamp
        ):

            p0 = world_to_pixel(*sl["p1"])
            p1 = world_to_pixel(*sl["p2"])

            cv2.line(
                img,
                p0,
                p1,
                255,
                2
            )

    return img

# ============================================================
# SAVE TOPOLOGY
# ============================================================

topology = cv2.imread(
    TOPOLOGY_PNG,
    cv2.IMREAD_GRAYSCALE
)

topology = cv2.resize(
    topology,
    (IMAGE_SIZE, IMAGE_SIZE)
)

cv2.imwrite(
    "dataset_png/topology.png",
    topology
)

# ============================================================
# LOAD FRAMES
# ============================================================

print("Loading frames...")

all_frames = list(
    iter_frames(TRAJ_ROOT)
)

total_frames = len(all_frames)

print(
    f"Total raw frames: {total_frames}"
)

max_offset = FRAME_SKIP * 3

count = 0

# ============================================================
# GENERATE DATASET
# ============================================================

for i in tqdm(
    range(total_frames - max_offset)
):

    if (
        MAX_FRAMES is not None
        and count >= MAX_FRAMES
    ):
        break

    f_t2 = all_frames[i]
    f_t1 = all_frames[i + FRAME_SKIP]
    f_t0 = all_frames[i + 2 * FRAME_SKIP]
    f_p1 = all_frames[i + 3 * FRAME_SKIP]

    ts_ms = int(
        f_t0.timestamp.timestamp() * 1000
    )

    out_dir = OUT_DIR / f"{ts_ms:015d}"
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    cv2.imwrite(
        str(out_dir / "dyn_box_t-2.png"),
        rasterize_box(f_t2)
    )

    cv2.imwrite(
        str(out_dir / "dyn_box_t-1.png"),
        rasterize_box(f_t1)
    )

    cv2.imwrite(
        str(out_dir / "dyn_box_t0.png"),
        rasterize_box(f_t0)
    )

    cv2.imwrite(
        str(out_dir / "dyn_box_t+1.png"),
        rasterize_box(f_p1)
    )

    stop_img = rasterize_stoplines(
        f_t0.timestamp.timestamp()
    )

    cv2.imwrite(
        str(out_dir / "stoplines.png"),
        stop_img
    )

    count += 1

print()
print(
    f"✅ Generated {count} samples"
)
print(
    f"dt = {DT_SECONDS}s"
)