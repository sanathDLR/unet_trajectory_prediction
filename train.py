import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from dataset import SceneEvolutionDataset
from model import ConvLSTM_UNet

# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "dataset_png"
TOPOLOGY_PATH = "dataset_png/topology.png"

BATCH_SIZE = 2
EPOCHS = 30
LR = 1e-4
VAL_SPLIT = 0.2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROLL_STEPS = 3

torch.backends.cudnn.benchmark = True

# ============================================================
# DATASET
# ============================================================

dataset = SceneEvolutionDataset(
    DATA_ROOT,
    TOPOLOGY_PATH
)

val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)

# ============================================================
# LOSSES
# ============================================================

def dice_loss(pred_logits, target, eps=1e-6):

    pred = torch.sigmoid(pred_logits)

    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=1)

    union = (
        pred.sum(dim=1)
        + target.sum(dim=1)
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        union + eps
    )

    return 1.0 - dice.mean()


bce_loss = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([15.0]).to(DEVICE)
)

# ============================================================
# IOU
# ============================================================

def compute_iou(pred_logits, target):

    pred = (torch.sigmoid(pred_logits) > 0.5).float()

    intersection = (
        pred * target
    ).sum(dim=(1,2,3))

    union = (
        (pred + target) > 0
    ).float().sum(dim=(1,2,3))

    iou = (intersection + 1e-6) / (union + 1e-6)

    return iou.mean().item()

# ============================================================
# MODEL
# ============================================================

model = ConvLSTM_UNet().to(DEVICE)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-5
)

best_val_loss = float("inf")

# ============================================================
# TRAIN
# ============================================================

for epoch in range(EPOCHS):

    # --------------------------------------------------
    # TRAIN
    # --------------------------------------------------

    model.train()

    train_loss = 0.0
    train_iou = 0.0

    for dyn_seq, static, future, frame_ids in train_loader:

        dyn_seq = dyn_seq.to(
            DEVICE,
            non_blocking=True
        )

        static = static.to(
            DEVICE,
            non_blocking=True
        )

        future = future.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad()

        current_dyn = dyn_seq.clone()

        loss = 0.0
        final_pred = None

        # ----------------------------------
        # rollout training
        # ----------------------------------

        for step in range(ROLL_STEPS):

            pred = model(
                current_dyn,
                static
            )

            bce = bce_loss(pred, future)
            dice = dice_loss(pred, future)

            loss += bce + dice

            final_pred = pred

            # ----------------------------------
            # autoregressive update
            # ----------------------------------

            next_frame = torch.sigmoid(pred)

            next_frame = (
                next_frame
                .unsqueeze(1)
            )  # (B,1,1,H,W)

            current_dyn = torch.cat(
                [
                    current_dyn[:, 1:],
                    next_frame.detach()
                ],
                dim=1
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        train_loss += loss.item()
        train_iou += compute_iou(
            final_pred,
            future
        )

    train_loss /= len(train_loader)
    train_iou /= len(train_loader)

    # --------------------------------------------------
    # VALIDATION
    # --------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_iou = 0.0

    with torch.no_grad():

        for dyn_seq, static, future, frame_ids in val_loader:

            dyn_seq = dyn_seq.to(DEVICE)
            static = static.to(DEVICE)
            future = future.to(DEVICE)

            current_dyn = dyn_seq.clone()

            loss = 0.0
            final_pred = None

            for step in range(ROLL_STEPS):

                pred = model(
                    current_dyn,
                    static
                )

                bce = bce_loss(pred, future)
                dice = dice_loss(pred, future)

                loss += bce + dice

                final_pred = pred

                next_frame = (
                    torch.sigmoid(pred)
                    .unsqueeze(1)
                )

                current_dyn = torch.cat(
                    [
                        current_dyn[:, 1:],
                        next_frame
                    ],
                    dim=1
                )

            val_loss += loss.item()

            val_iou += compute_iou(
                final_pred,
                future
            )

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)

    print(
        f"Epoch {epoch+1:03d}/{EPOCHS} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train IoU {train_iou:.4f} | "
        f"Val Loss {val_loss:.4f} | "
        f"Val IoU {val_iou:.4f}"
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            model.state_dict(),
            "best_convlstm_unet.pt"
        )

        print("✅ Best model saved")

print("\nTraining complete.")