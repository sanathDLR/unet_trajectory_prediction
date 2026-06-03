import torch
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
import numpy as np


class SceneEvolutionDataset(Dataset):
    def __init__(self, root_dir, topology_path, max_samples=None):

        self.root_dir = Path(root_dir)

        # ---------------------------------
        # Load frame folders
        # ---------------------------------
        self.frame_dirs = sorted(
            self.root_dir.glob("frames/*"),
            key=lambda p: int(p.name)
        )

        if max_samples is not None:
            self.frame_dirs = self.frame_dirs[:max_samples]

        # ---------------------------------
        # Extract frame IDs
        # ---------------------------------
        self.frame_ids = [
            int(p.name)
            for p in self.frame_dirs
        ]

        # ---------------------------------
        # Load topology once
        # ---------------------------------
        topo = Image.open(
            topology_path
        ).convert("L")

        self.topology = torch.from_numpy(
            np.array(
                topo,
                dtype=np.float32
            ) / 255.0
        ).float()

        # ---------------------------------
        # Required files
        # ---------------------------------
        required = [
            "dyn_box_t-2.png",
            "dyn_box_t-1.png",
            "dyn_box_t0.png",
            "dyn_box_t+1.png",
            "stoplines.png",
        ]

        valid_dirs = []
        valid_ids = []

        for d, fid in zip(
            self.frame_dirs,
            self.frame_ids
        ):

            if all(
                (d / f).exists()
                for f in required
            ):
                valid_dirs.append(d)
                valid_ids.append(fid)

        self.frame_dirs = valid_dirs
        self.frame_ids = valid_ids

        print(
            f"Dataset initialized with "
            f"{len(self.frame_dirs)} samples."
        )

    def __len__(self):
        return len(self.frame_dirs)

    # ---------------------------------
    # Image loader
    # ---------------------------------

    def load_gray(self, path):

        img = Image.open(path).convert("L")

        arr = np.array(
            img,
            dtype=np.float32
        ) / 255.0

        return torch.from_numpy(arr).float()

    def __getitem__(self, idx):

        d = self.frame_dirs[idx]
        frame_id = self.frame_ids[idx]

        # =====================================
        # TEMPORAL INPUT
        # =====================================
        dyn_seq = torch.stack([
            self.load_gray(
                d / "dyn_box_t-2.png"
            ),
            self.load_gray(
                d / "dyn_box_t-1.png"
            ),
            self.load_gray(
                d / "dyn_box_t0.png"
            ),
        ], dim=0)

        # (T,H,W) -> (T,1,H,W)
        dyn_seq = dyn_seq.unsqueeze(1)

        # =====================================
        # STATIC INPUT
        # =====================================

        stoplines = self.load_gray(
            d / "stoplines.png"
        )

        static = torch.stack([
            self.topology,
            stoplines
        ], dim=0)

        # =====================================
        # TARGET
        # =====================================

        future = self.load_gray(
            d / "dyn_box_t+1.png"
        ).unsqueeze(0)

        # =====================================
        # RETURN
        # =====================================

        return (
            dyn_seq,   # (3,1,H,W)
            static,    # (2,H,W)
            future,    # (1,H,W)
            frame_id
        )