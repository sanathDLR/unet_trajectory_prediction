import torch
import numpy as np
import imageio
import cv2

from dataset import SceneEvolutionDataset
from model import ConvLSTM_UNet

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "dataset_png"
TOPOLOGY_PATH = "dataset_png/topology.png"
MODEL_PATH = "best_convlstm_unet.pt"

START_FRAME_ID = 1695530114766

ROLL_SECONDS = 5.0
DT = 0.20
ROLL_STEPS = int(ROLL_SECONDS / DT)

THRESH = 0.5
GIF_PATH = "rollout.gif"
FPS = int(1.0 / DT)

FRAME_SKIP = 4  # IMPORTANT

# ============================================================
# IOU FUNCTION
# ============================================================

def compute_iou(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return intersection / union

# ============================================================
# LOAD DATASET
# ============================================================

dataset = SceneEvolutionDataset(DATA_ROOT, TOPOLOGY_PATH)
frame_ids = [int(p.name) for p in dataset.frame_dirs]

if START_FRAME_ID not in frame_ids:
    raise ValueError("Frame ID not found")

start_idx = frame_ids.index(START_FRAME_ID)

# ============================================================
# LOAD MODEL
# ============================================================

model = ConvLSTM_UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ============================================================
# INITIAL STATE
# ============================================================

dyn_seq, static, _ = dataset[start_idx]

stoplines = static[1].numpy()

dyn_seq = dyn_seq.unsqueeze(0).to(DEVICE)
static  = static.unsqueeze(0).to(DEVICE)

current_dyn = dyn_seq.clone()

frames = []
ious = []

# ============================================================
# ROLLOUT
# ============================================================

with torch.no_grad():

    for step in range(ROLL_STEPS):

        pred_logits = model(current_dyn, static)
        pred = torch.sigmoid(pred_logits)

        pred_frame = pred[0, 0]
        pred_np = pred_frame.cpu().numpy()

        pred_bin = (pred_np > THRESH).astype(np.uint8)

        # ----------------------------
        # Ground truth (FIXED)
        # ----------------------------
        if step == 0:
            gt = (dataset[start_idx][2][0].numpy() > 0.5).astype(np.uint8)
        else:
            gt_idx = start_idx + step * FRAME_SKIP
            if gt_idx < len(dataset):
                _, _, future = dataset[gt_idx]
                gt = (future[0].numpy() > 0.5).astype(np.uint8)
            else:
                gt = None

        # ----------------------------
        # IoU calculation
        # ----------------------------
        if gt is not None:
            iou = compute_iou(pred_bin, gt)
            ious.append(iou)
        else:
            ious.append(np.nan)

        # ----------------------------
        # Create RGB frame
        # ----------------------------
        H, W = pred_bin.shape
        frame = np.zeros((H, W, 3), dtype=np.uint8)

        # Red = prediction
        frame[pred_bin == 1] = [255, 0, 0]

        if gt is not None:
            frame[gt == 1] = [0, 255, 0]

            overlap = (gt == 1) & (pred_bin == 1)
            frame[overlap] = [255, 255, 0]

        # Stoplines
        blue_mask = stoplines > 0.5
        frame[blue_mask] = (
            frame[blue_mask] * 0.5 + np.array([0, 0, 255]) * 0.5
        ).astype(np.uint8)

        # Add text
        text = f"t+{step+1} ({(step+1)*DT:.1f}s)"
        cv2.putText(frame, text, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

        frames.append(frame)

        # ----------------------------
        # Update sequence
        # ----------------------------
        new_frame = pred_frame[None, None, None, :, :]

        current_dyn = torch.cat([
            current_dyn[:, 1:],
            new_frame
        ], dim=1)

# ============================================================
# SAVE GIF
# ============================================================

imageio.mimsave(GIF_PATH, frames, fps=FPS)

# ============================================================
# PRINT IOU RESULTS
# ============================================================

print(f"✅ GIF saved as {GIF_PATH}")

print("\n===== IoU over rollout =====")

for i, val in enumerate(ious):
    if not np.isnan(val):
        print(f"Step {i+1} ({(i+1)*DT:.1f}s): IoU = {val:.4f}")
    else:
        print(f"Step {i+1}: IoU = N/A")

valid_ious = [v for v in ious if not np.isnan(v)]

if len(valid_ious) > 0:
    mean_iou = np.mean(valid_ious)
    print(f"\nAverage IoU over {len(valid_ious)} steps: {mean_iou:.4f}")

print("\nLegend:")
print("🔴 Red     = Prediction")
print("🟢 Green   = Ground Truth")
print("🟡 Yellow  = Overlap")
print("🔵 Blue    = Stoplines")