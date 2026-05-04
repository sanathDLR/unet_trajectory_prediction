import torch
import numpy as np
import matplotlib.pyplot as plt

from dataset import SceneEvolutionDataset
from model import ConvLSTM_UNet

# ============================================================
# CONFIG
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "dataset_png"
TOPOLOGY_PATH = "dataset_png/topology.png"
MODEL_PATH = "best_convlstm_unet.pt"

TARGET_FRAME_ID = 1695514274466
THRESH = 0.5
SAVE_FIG = "prediction_visualization.png"

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

dataset = SceneEvolutionDataset(
    root_dir=DATA_ROOT,
    topology_path=TOPOLOGY_PATH
)

frame_ids = [int(p.name) for p in dataset.frame_dirs]

if TARGET_FRAME_ID not in frame_ids:
    raise ValueError("Frame ID not found in dataset.")

idx = frame_ids.index(TARGET_FRAME_ID)

dyn_seq, static, Y = dataset[idx]

dyn_seq = dyn_seq.to(DEVICE)
static  = static.to(DEVICE)

# ============================================================
# LOAD MODEL
# ============================================================

model = ConvLSTM_UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ============================================================
# PREDICT
# ============================================================

with torch.no_grad():
    pred_logits = model(
        dyn_seq.unsqueeze(0),
        static.unsqueeze(0)
    )
    pred = torch.sigmoid(pred_logits)[0, 0]

pred_bin = (pred > THRESH).cpu().numpy().astype(np.uint8)
gt = (Y[0].numpy() > 0.5).astype(np.uint8)

# ============================================================
# COMPUTE IOU
# ============================================================

iou = compute_iou(pred_bin, gt)

# ============================================================
# PLOT
# ============================================================

fig, axs = plt.subplots(3, 3, figsize=(12, 12))

# ----------------------------
# Row 1: Input dynamics
# ----------------------------

axs[0,0].imshow(dyn_seq[0,0].cpu(), cmap="gray")
axs[0,0].set_title("Input dyn_t-2")

axs[0,1].imshow(dyn_seq[1,0].cpu(), cmap="gray")
axs[0,1].set_title("Input dyn_t-1")

axs[0,2].imshow(dyn_seq[2,0].cpu(), cmap="gray")
axs[0,2].set_title("Input dyn_t0")

# ----------------------------
# Row 2: GT vs Prediction
# ----------------------------

axs[1,0].imshow(gt, cmap="gray")
axs[1,0].set_title("GT t+1")

axs[1,1].imshow(pred.cpu(), cmap="hot")
axs[1,1].set_title("Pred Prob")

axs[1,2].imshow(pred_bin, cmap="gray")
axs[1,2].set_title("Pred Binary")

# ----------------------------
# Row 3: Overlay + Stoplines
# ----------------------------

overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)

# Green = GT
overlay[gt == 1] = [0, 255, 0]

# Red = prediction
overlay[pred_bin == 1] = [255, 0, 0]

# Yellow = overlap
overlap = (gt == 1) & (pred_bin == 1)
overlay[overlap] = [255, 255, 0]

# 🔵 Stoplines
stoplines = static[1].cpu().numpy()
blue_mask = stoplines > 0.5
overlay[blue_mask] = (
    overlay[blue_mask] * 0.5 + np.array([0, 0, 255]) * 0.5
).astype(np.uint8)

axs[2,0].imshow(overlay)
axs[2,0].set_title(f"Overlay + Stoplines\nIoU = {iou:.4f}")

# Empty slots
axs[2,1].axis("off")
axs[2,2].axis("off")

# ----------------------------
# Clean layout
# ----------------------------

for ax in axs.flat:
    ax.axis("off")

plt.tight_layout()
plt.savefig(SAVE_FIG, dpi=150)
plt.close()

# ============================================================
# PRINT RESULTS
# ============================================================

print(f"✅ Visualization saved as {SAVE_FIG}")
print(f"\nIoU (t+1): {iou:.4f}")

print("\nLegend:")
print("🟢 Green   = Ground Truth")
print("🔴 Red     = Prediction")
print("🟡 Yellow  = Overlap")
print("🔵 Blue    = Stoplines")