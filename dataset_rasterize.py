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
# MAX_FRAMES = 50

DT_SECONDS = 0.2
ORIGINAL_DT = 0.05
FRAME_SKIP = int(DT_SECONDS / ORIGINAL_DT)  # = 4

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
TOPOLOGY_PNG   = "lane_topology.png"

STOP_STATES = {3, 4}

# ============================================================
# LOAD BOUNDS
# ============================================================

first_csv = sorted(TRAJ_ROOT.glob("trajectories_*.csv"))[0]
bounds_df = pd.read_csv(first_csv, nrows=1000)
bounds = load_or_compute_bounds(bounds_df)

left, right = bounds["left"], bounds["right"]
bottom, top = bounds["bottom"], bounds["top"]

scale_x = IMAGE_SIZE / (right - left)
scale_y = IMAGE_SIZE / (top - bottom)

def world_to_pixel(e, n):
    px = int((e - left) * scale_x)
    py = int((top - n) * scale_y)
    return np.clip(px, 0, IMAGE_SIZE-1), np.clip(py, 0, IMAGE_SIZE-1)

# ============================================================
# TRAFFIC LIGHTS
# ============================================================

with open(STOP_LINE_JSON, "r") as f:
    STOP_LINES = json.load(f)["stop_lines"]

def load_traffic_lights():
    csvs = glob.glob(str(TL_DIR / "**/*.csv"), recursive=True)
    dfs = [pd.read_csv(c) for c in csvs]
    df = pd.concat(dfs, ignore_index=True)

    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], format="mixed", utc=True)
        .astype("int64") / 1e9
    )

    tl_states = {}
    for tl_id, g in df.groupby("id"):
        times = g["timestamp"].to_numpy()
        states = g["state"].to_numpy()
        order = np.argsort(times)
        tl_states[int(tl_id)] = (times[order], states[order])

    return tl_states

TL_STATES = load_traffic_lights()

def is_stop_required(sl_id, timestamp):
    if sl_id not in TL_STATES:
        return False
    times, states = TL_STATES[sl_id]
    idx = np.searchsorted(times, timestamp, side="right") - 1
    return idx >= 0 and states[idx] in STOP_STATES

# ============================================================
# RASTER FUNCTIONS
# ============================================================

def oriented_box_corners(x, y, yaw_deg, length, width):
    yaw = math.radians(yaw_deg)
    dx, dy = length/2, width/2

    corners = np.array([[ dx, dy], [ dx,-dy], [-dx,-dy], [-dx, dy]])
    R = np.array([[math.cos(yaw), -math.sin(yaw)],
                  [math.sin(yaw),  math.cos(yaw)]])

    return corners @ R.T + np.array([x, y])

def rasterize_box(frame: Frame):
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    for ag in frame.agents:
        corners = oriented_box_corners(
            ag.easting, ag.northing,
            ag.yaw_deg, ag.length_m, ag.width_m
        )
        pts = np.array([world_to_pixel(x, y) for x, y in corners], np.int32)
        cv2.fillPoly(img, [pts], 255)

    return img

def rasterize_gaussian(frame: Frame):
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for ag in frame.agents:
        cx, cy = world_to_pixel(ag.easting, ag.northing)

        sigma_x = ag.length_m * scale_x * 0.3
        sigma_y = ag.width_m  * scale_y * 0.3

        size = int(max(sigma_x, sigma_y) * 3)
        xs = np.arange(-size, size)
        ys = np.arange(-size, size)
        X, Y = np.meshgrid(xs, ys)

        gaussian = np.exp(-(X**2/(2*sigma_x**2) + Y**2/(2*sigma_y**2)))

        x0, x1 = cx-size, cx+size
        y0, y1 = cy-size, cy+size

        if x1 < 0 or y1 < 0 or x0 >= IMAGE_SIZE or y0 >= IMAGE_SIZE:
            continue

        gx0 = max(0, -x0)
        gy0 = max(0, -y0)
        gx1 = min(2*size, IMAGE_SIZE-x0)
        gy1 = min(2*size, IMAGE_SIZE-y0)

        ix0 = max(0, x0)
        iy0 = max(0, y0)
        ix1 = min(IMAGE_SIZE, x1)
        iy1 = min(IMAGE_SIZE, y1)

        img[iy0:iy1, ix0:ix1] = np.maximum(
            img[iy0:iy1, ix0:ix1],
            gaussian[gy0:gy1, gx0:gx1]
        )

    return (img * 255).astype(np.uint8)

def rasterize_stoplines(timestamp):
    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for sl in STOP_LINES:
        if is_stop_required(sl["id"], timestamp):
            p0 = world_to_pixel(*sl["p1"])
            p1 = world_to_pixel(*sl["p2"])
            cv2.line(img, p0, p1, 255, 2)
    return img

# ============================================================
# SAVE TOPOLOGY
# ============================================================

topo = cv2.imread(TOPOLOGY_PNG, cv2.IMREAD_GRAYSCALE)
topo = cv2.resize(topo, (IMAGE_SIZE, IMAGE_SIZE))
cv2.imwrite("dataset_png/topology.png", topo)

# ============================================================
# MAIN LOOP
# ============================================================

all_frames = list(iter_frames(TRAJ_ROOT))
total_frames = len(all_frames)

max_offset = FRAME_SKIP * 3
count = 0

for i in tqdm(range(total_frames - max_offset)):

    # if count >= MAX_FRAMES:
    #     break

    frames = [
        all_frames[i],
        all_frames[i + FRAME_SKIP],
        all_frames[i + 2*FRAME_SKIP],
        all_frames[i + 3*FRAME_SKIP],
    ]

    ts_ms = int(frames[2].timestamp.timestamp() * 1000)
    out = OUT_DIR / f"{ts_ms:015d}"
    out.mkdir(parents=True, exist_ok=True)

    names = ["t-2", "t-1", "t0", "t+1"]

    for f, name in zip(frames, names):
        cv2.imwrite(out / f"dyn_box_{name}.png", rasterize_box(f))
        cv2.imwrite(out / f"dyn_gauss_{name}.png", rasterize_gaussian(f))

    stop_img = rasterize_stoplines(frames[2].timestamp.timestamp())
    cv2.imwrite(out / "stoplines.png", stop_img)

    count += 1

print(f"\n✅ Generated {count} samples with box + gaussian channels")