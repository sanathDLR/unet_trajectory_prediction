import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dataset import SceneEvolutionDataset
from model import ConvLSTM_UNet

# =====================
# CONFIG
# =====================

DATA_ROOT = "dataset_png"
TOPOLOGY  = "dataset_png/topology.png"

BATCH_SIZE = 4
EPOCHS = 3
LR = 5e-5
VAL_SPLIT = 0.2

ROLL_STEPS = 3  # 🔥 autoregressive steps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True  # 🔥 speed boost

# =====================
# DATASET
# =====================

dataset = SceneEvolutionDataset(DATA_ROOT, TOPOLOGY)

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# =====================
# LOSS FUNCTIONS
# =====================

def dice_loss(pred_logits, targets, eps=1e-6):
    pred = torch.sigmoid(pred_logits)

    pred = pred.contiguous().view(pred.size(0), -1)
    targets = targets.contiguous().view(targets.size(0), -1)

    intersection = (pred * targets).sum(dim=1)
    union = pred.sum(dim=1) + targets.sum(dim=1)

    dice = (2.0 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


bce_loss_fn = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([15.0]).to(DEVICE)
)

# =====================
# MODEL
# =====================

model = ConvLSTM_UNet().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val = float("inf")

# =====================
# HELPER: rebuild gaussian
# =====================

def build_gaussian(box):
    """
    box: (B,1,H,W)
    returns: (B,1,H,W)
    """
    gauss = F.avg_pool2d(box, kernel_size=7, stride=1, padding=3)

    # normalize
    max_val = gauss.amax(dim=(2,3), keepdim=True) + 1e-6
    gauss = gauss / max_val

    return gauss


# =====================
# TRAIN LOOP
# =====================

for epoch in range(EPOCHS):

    # -------- TRAIN --------
    model.train()
    train_loss = 0

    for dyn_seq, static, Y, frame_ids in train_loader:

        dyn_seq = dyn_seq.to(DEVICE, non_blocking=True)   # (B,T,2,H,W)
        static  = static.to(DEVICE, non_blocking=True)
        Y       = Y.to(DEVICE, non_blocking=True)         # (B,1,H,W)

        optimizer.zero_grad()

        current_dyn = dyn_seq.clone()
        loss = 0

        # 🔥 AUTOREGRESSIVE ROLLOUT
        for step in range(ROLL_STEPS):

            pred = model(current_dyn, static)  # (B,1,H,W)

            # ---- Loss ----
            bce = bce_loss_fn(pred, Y)
            dice = dice_loss(pred, Y)

            loss += bce + dice

            # ---- Rebuild input (CRITICAL FIX) ----
            box = pred
            gauss = build_gaussian(box)

            new_frame = torch.cat([box, gauss], dim=1)  # (B,2,H,W)
            new_frame = new_frame.unsqueeze(1)          # (B,1,2,H,W)

            current_dyn = torch.cat([
                current_dyn[:, 1:],     # drop oldest
                new_frame.detach()      # avoid gradient explosion
            ], dim=1)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    # -------- VALIDATION --------
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for dyn_seq, static, Y, frame_ids in val_loader:

            dyn_seq = dyn_seq.to(DEVICE, non_blocking=True)
            static  = static.to(DEVICE, non_blocking=True)
            Y       = Y.to(DEVICE, non_blocking=True)

            current_dyn = dyn_seq.clone()
            loss = 0

            for step in range(ROLL_STEPS):

                pred = model(current_dyn, static)

                bce = bce_loss_fn(pred, Y)
                dice = dice_loss(pred, Y)

                loss += bce + dice

                # ---- rebuild channels ----
                box = pred
                gauss = build_gaussian(box)

                new_frame = torch.cat([box, gauss], dim=1)
                new_frame = new_frame.unsqueeze(1)

                current_dyn = torch.cat([
                    current_dyn[:, 1:],
                    new_frame
                ], dim=1)

            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS} | "
          f"Train: {train_loss:.4f} | "
          f"Val: {val_loss:.4f}")

    # -------- SAVE BEST --------
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_convlstm_unet.pt")
        print("✅ Best model saved")

print("Training complete.")