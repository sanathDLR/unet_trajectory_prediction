# utils/render_utils.py
from __future__ import annotations
import math, numpy as np
from config import IMAGE_SIZE, MARGIN, GAUSS_GAIN, GAUSS_MIN_PX
import cv2

# -------------------------------------------------------------------------
def world_to_pixel(e, n, bounds, image_size=IMAGE_SIZE, margin=MARGIN):
    w, h = image_size
    x = (e - bounds["left"])  / (bounds["right"] - bounds["left"])
    y = (n - bounds["bottom"])/(bounds["top"]   - bounds["bottom"])
    px = margin + x * (w - 2*margin)
    py = margin + (1 - y) * (h - 2*margin)
    return px, py

# -------------------------------------------------------------------------
def sigma_from_dims(length_m, width_m,
                    gain=GAUSS_GAIN, min_px=GAUSS_MIN_PX):
    return max(length_m*gain, min_px), max(width_m*gain, min_px)

def draw_gaussian(canvas, cx, cy, yaw, sig_x, sig_y):
    h, w = canvas.shape
    ys = np.arange(h)[:, None]
    xs = np.arange(w)[None, :]
    cos, sin = math.cos(-yaw), math.sin(-yaw)  # img y down
    dx = xs - cx
    dy = ys - cy
    bx = cos*dx + sin*dy
    by = -sin*dx + cos*dy
    g = np.exp(-0.5*((bx/sig_x)**2 + (by/sig_y)**2))
    np.maximum(canvas, g, out=canvas)    # additive can saturate; max looks good

# -------------------------------------------------------------------------
def draw_box(img, cx, cy, length_px, width_px, yaw):
    cos, sin = math.cos(-yaw), math.sin(-yaw)
    dx, dy = length_px/2, width_px/2
    corners = np.array([[-dx,-dy],[dx,-dy],[dx,dy],[-dx,dy]])
    rot = np.array([[cos,-sin],[sin,cos]])
    rc  = (corners @ rot.T) + np.array([cx, cy])
    cv2.fillPoly(img, [rc.astype(np.int32)], color=(255,255,255))

# ---------------------------------------------------------------------------
#  Differentiable single oriented Gaussian  (Torch)
# ---------------------------------------------------------------------------
from typing import Tuple, Optional
import torch
from torch import Tensor
import math

_pixel_grid_cache: Optional[Tuple[Tensor, Tensor]] = None

def _pixel_grid_torch(device: torch.device, h: int, w: int) -> Tuple[Tensor, Tensor]:
    """
    Cached X,Y pixel-index tensors (float64).  Re-used across calls.
    """
    global _pixel_grid_cache
    if _pixel_grid_cache is None or _pixel_grid_cache[0].shape != (h, w):
        ys = torch.arange(h, dtype=torch.float64).view(-1, 1).repeat(1, w)
        xs = torch.arange(w, dtype=torch.float64).view(1, -1).repeat(h, 1)
        _pixel_grid_cache = (xs, ys)          # cache on CPU
    xs, ys = _pixel_grid_cache
    return xs.to(device), ys.to(device)

def render_oriented_gaussian_torch(
    e_px: Tensor,
    n_px: Tensor,
    yaw_rad: Tensor,
    sigma_long: float,
    sigma_lat: float,
    h: int,
    w: int,
    device: torch.device,
) -> Tensor:
    """
    Render a single oriented Gaussian footprint as a float64 tensor (H×W, [0,1]).

    Parameters
    ----------
    e_px, n_px : float/1-elem Tensor – centre in *pixel* coords
    yaw_rad    : heading in *radians* (world frame; +X east, +Y north)
    sigma_long : std dev along vehicle length  (pixels)
    sigma_lat  : std dev along vehicle width   (pixels)
    h, w       : output resolution
    device     : CUDA / CPU
    """
    xs, ys = _pixel_grid_torch(device, h, w)       # (H,W) float64
    yaw_img = -yaw_rad                             # flip for image Y-axis down
    cos_t, sin_t = torch.cos(yaw_img), torch.sin(yaw_img)

    dx = xs - e_px
    dy = ys - n_px
    bx =  cos_t * dx + sin_t * dy                  # body-long axis
    by = -sin_t * dx + cos_t * dy                  # body-lat  axis

    inv2sx2 = 0.5 / (sigma_long ** 2)
    inv2sy2 = 0.5 / (sigma_lat ** 2)
    g = torch.exp(-(bx * bx) * inv2sx2 - (by * by) * inv2sy2)
    return torch.clamp(g, 0.0, 1.0)



# ---------------------------------------------------------------------------
#  render_gaussians_steps_torch
# ---------------------------------------------------------------------------
def render_gaussians_steps_torch(
    states_px_yaw: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    sigmas: tuple[float, float],
    h: int,
    w: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Render N oriented-gaussian channels (one per time-step).

    Parameters
    ----------
    states_px_yaw : list[(px, py, yaw_rad)]  – px / py may be tensors
    sigmas        : (sigma_long_px, sigma_lat_px)
    h, w          : output resolution (pixels)
    device        : CUDA / CPU target

    Returns
    -------
    torch.Tensor  – shape (N, H, W), dtype float64, values ∈ [0,1]
    """
    sx, sy = sigmas
    chans = [
        render_oriented_gaussian_torch(px, py, yaw,
                                       sigma_long=sx, sigma_lat=sy,
                                       h=h, w=w, device=device)
        for (px, py, yaw) in states_px_yaw
    ]
    return torch.stack(chans, dim=0)  # (N,H,W)
