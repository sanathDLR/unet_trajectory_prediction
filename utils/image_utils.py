# utils/image_utils.py
from pathlib import Path
from typing import Tuple, Optional

from PIL import Image
import numpy as np
import torch
from torchvision import transforms


def save_rgb(img: np.ndarray, path: Path) -> None:
    """Expect img uint8 H×W×3."""
    Image.fromarray(img).save(path)

def load_grayscale_tensor(
    path: Path | str,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load a greyscale PNG/JPEG and return a 1×H×W tensor in [0,1].

    * Keeps dtype & device configurable (defaults float32 / CPU).
    * No automatic resizing — caller decides.
    """
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0     # H×W float32
    ten = torch.tensor(arr, dtype=dtype, device=device).unsqueeze(0)  # 1×H×W
    return ten


# ---------------------------------------------------------------------------
# 2.  Lane topology map loader & resizer  →  Tensor  (1×H×W, float32, [0,1])
# ---------------------------------------------------------------------------
def load_and_resize_lane_tensor(
    path: Path | str,
    target_size: Tuple[int, int],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Load a greyscale lane-topology image, resize to target (W,H),
    and return a 1×H×W float tensor in [0,1].

    * Uses bilinear interpolation via torchvision transforms.
    """
    img_pil = Image.open(path).convert("L")
    to_tensor = transforms.ToTensor()        # → 1×H×W in [0,1] float32
    ten = to_tensor(img_pil)                 # float32 CPU
    if ten.shape[-2:] != target_size[::-1]:  # (H,W) vs (W,H)
        resize = transforms.Resize(target_size[::-1], interpolation=transforms.InterpolationMode.BILINEAR)
        ten = resize(ten)
    return ten.to(device=device, dtype=dtype)