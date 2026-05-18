import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


class SceneEvolutionDataset(Dataset):
    def __init__(self, root_dir, topology_path, max_samples=None):

        self.root_dir = Path(root_dir)

        # ---------------------------------
        # Load all frame folders (sorted by timestamp)
        # ---------------------------------
        self.frame_dirs = sorted(
            self.root_dir.glob("frames/*"),
            key=lambda p: int(p.name)
        )

        if max_samples is not None:
            self.frame_dirs = self.frame_dirs[:max_samples]

        # ---------------------------------
        # Extract frame IDs (timestamps)
        # ---------------------------------
        self.frame_ids = [int(p.name) for p in self.frame_dirs]

        # ---------------------------------
        # Load topology (static)
        # ---------------------------------
        topo = Image.open(topology_path).convert("L")
        self.topology = torch.from_numpy(
            np.array(topo, dtype=np.float32) / 255.0
        ).float()

        # ---------------------------------
        # Filter valid samples
        # ---------------------------------
        required = [
            # box
            "dyn_box_t-2.png",
            "dyn_box_t-1.png",
            "dyn_box_t0.png",
            "dyn_box_t+1.png",

            # gaussian
            "dyn_gauss_t-2.png",
            "dyn_gauss_t-1.png",
            "dyn_gauss_t0.png",

            # static
            "stoplines.png",
        ]

        valid_dirs = []
        valid_ids = []

        for d, fid in tqdm(zip(self.frame_dirs, self.frame_ids), total=len(self.frame_dirs), desc="Loading dataset"):
            if all((d / f).exists() for f in required):
                valid_dirs.append(d)
                valid_ids.append(fid)

        self.frame_dirs = valid_dirs
        self.frame_ids = valid_ids

        print(f"Dataset initialized with {len(self.frame_dirs)} samples.")

    def __len__(self):
        return len(self.frame_dirs)

    def load_gray(self, path):
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).float()

    def __getitem__(self, idx):

        d = self.frame_dirs[idx]
        frame_id = self.frame_ids[idx]

        # =========================================
        # TEMPORAL INPUT (T=3, C=2)
        # =========================================
        dyn_seq = []

        for t in ["t-2", "t-1", "t0"]:
            box   = self.load_gray(d / f"dyn_box_{t}.png")
            gauss = self.load_gray(d / f"dyn_gauss_{t}.png")

            dyn = torch.stack([box, gauss], dim=0)  # (2, H, W)
            dyn_seq.append(dyn)

        dyn_seq = torch.stack(dyn_seq, dim=0)  # (T=3, C=2, H, W)

        # =========================================
        # STATIC INPUT
        # =========================================
        stoplines = self.load_gray(d / "stoplines.png")

        static = torch.stack([
            self.topology,
            stoplines
        ], dim=0)  # (2, H, W)

        # =========================================
        # TARGET (BOX ONLY)
        # =========================================
        future = self.load_gray(d / "dyn_box_t+1.png").unsqueeze(0)

        # =========================================
        # RETURN
        # =========================================
        return dyn_seq, static, future, frame_id