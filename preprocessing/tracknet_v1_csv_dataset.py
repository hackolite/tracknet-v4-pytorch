"""
TrackNet V1 CSV Dataset Loader

Reads frames and label.csv annotation files from a dataset laid out as:

    datasets/trackNet/
    ├── game1/
    │   ├── Clip1/
    │   │   ├── 0000.jpg
    │   │   ├── 0001.jpg
    │   │   └── label.csv
    │   └── Clip2/
    │       └── ...
    └── game2/
        └── ...

label.csv format::

    file name,visibility,x-coordinate,y-coordinate,status
    0000.jpg,1,599,423,0
    0001.jpg,1,601,406,0
    ...

Coordinates are given in the original image space and are scaled automatically
to 512×288 when generating heatmaps.

Output Format:
    - inputs:  (9, 288, 512) — 3 RGB frames concatenated, normalised to [0, 1]
    - heatmaps: (3, 288, 512) — 3 Gaussian heatmaps generated from coordinates
"""

import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

TARGET_WIDTH = 512
TARGET_HEIGHT = 288
HEATMAP_SIGMA = 3.0


def generate_heatmap(cx, cy, width=TARGET_WIDTH, height=TARGET_HEIGHT, sigma=HEATMAP_SIGMA):
    """Return a (1, H, W) float32 tensor with a 2-D Gaussian centred at (cx, cy)."""
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(xs, ys)
    heatmap = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2.0 * sigma ** 2))
    return torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)


class TrackNetV1CSVDataset(Dataset):
    """PyTorch Dataset for TrackNet-style datasets with per-clip label.csv files.

    Reads frames from ``game*/Clip*/*.jpg`` and generates Gaussian heatmaps
    on-the-fly from the (x, y, visibility) annotations in each
    ``game*/Clip*/label.csv``.  Frames are resized to 512×288; coordinates are
    scaled proportionally from the original image dimensions.
    """

    def __init__(self, root_dir, sigma=HEATMAP_SIGMA):
        """
        Args:
            root_dir: Root directory containing ``game*/`` sub-directories.
            sigma:    Standard deviation for the Gaussian heatmap (default 3.0).
        """
        self.root_dir = Path(root_dir)
        self.sigma = sigma
        self.transform = transforms.Compose([
            transforms.Resize((TARGET_HEIGHT, TARGET_WIDTH)),
            transforms.ToTensor(),
        ])
        self.data_items = self._scan_dataset()

    # ------------------------------------------------------------------
    # Dataset scanning
    # ------------------------------------------------------------------

    def _scan_dataset(self):
        game_dirs = sorted(d for d in self.root_dir.iterdir() if d.is_dir())
        if not game_dirs:
            raise FileNotFoundError(
                f"No game directories found under {self.root_dir}. "
                f"Expected structure: {self.root_dir}/game*/Clip*/label.csv"
            )
        print(f"Scanning {len(game_dirs)} game folder(s)…")

        items = []
        for game_dir in game_dirs:
            clip_dirs = sorted(d for d in game_dir.iterdir() if d.is_dir())
            for clip_dir in clip_dirs:
                label_file = clip_dir / "Label.csv"
                if not label_file.exists():
                    continue
                items.extend(self._process_clip(clip_dir, label_file))

        if not items:
            raise RuntimeError(
                f"No valid samples found under {self.root_dir}. "
                "Check that each Clip directory contains a label.csv file and at "
                "least 3 matching .jpg frames."
            )
        print(f"Found {len(items)} valid sample(s)")
        return items

    def _process_clip(self, clip_dir, label_file):
        labels = self._read_labels(label_file)
        if not labels:
            return []

        # Keep only images whose filenames appear in the label CSV
        img_files = sorted(
            [f for f in clip_dir.glob("*.jpg") if f.name in labels],
            key=lambda p: int(p.stem),
        )

        if len(img_files) < 3:
            return []

        # Determine original image size once for coordinate scaling
        orig_w, orig_h = self._get_image_size(img_files[0])
        scale_x = TARGET_WIDTH / orig_w
        scale_y = TARGET_HEIGHT / orig_h

        return [
            {
                "inputs": [str(img_files[i + j]) for j in range(3)],
                "labels": [labels[img_files[i + j].name] for j in range(3)],
                "scale": (scale_x, scale_y),
                "game": clip_dir.parent.name,
                "clip": clip_dir.name,
                "idx": i,
            }
            for i in range(len(img_files) - 2)
        ]

    @staticmethod
    def _read_labels(label_file):
        """Parse a label.csv file into a dict keyed by filename."""
        labels = {}
        try:
            with open(label_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row["file name"]
                    visibility = int(row["visibility"])
                    x = float(row["x-coordinate"]) if visibility else 0.0
                    y = float(row["y-coordinate"]) if visibility else 0.0
                    labels[fname] = (visibility, x, y)
        except Exception as exc:
            print(f"Warning: could not read {label_file}: {exc}")
        return labels

    @staticmethod
    def _get_image_size(path):
        """Return (width, height) of the image at *path*, with a safe fallback."""
        try:
            with Image.open(path) as img:
                return img.size  # PIL returns (width, height)
        except Exception:
            return (1280, 720)  # typical TrackNet source resolution

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def _load_image(self, image_path):
        try:
            return self.transform(Image.open(image_path).convert("RGB"))
        except Exception:
            return torch.zeros(3, TARGET_HEIGHT, TARGET_WIDTH)

    def _make_heatmap(self, visibility, x, y, scale_x, scale_y):
        if not visibility:
            return torch.zeros(1, TARGET_HEIGHT, TARGET_WIDTH)
        cx = max(0.0, min(TARGET_WIDTH - 1, x * scale_x))
        cy = max(0.0, min(TARGET_HEIGHT - 1, y * scale_y))
        return generate_heatmap(cx, cy, sigma=self.sigma)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        """
        Returns:
            inputs:   (9, 288, 512) — 3 RGB frames concatenated, values in [0, 1]
            heatmaps: (3, 288, 512) — 3 Gaussian heatmaps, values in [0, 1]
        """
        item = self.data_items[idx]
        scale_x, scale_y = item["scale"]

        inputs = torch.cat([self._load_image(p) for p in item["inputs"]], dim=0)
        heatmaps = torch.cat(
            [self._make_heatmap(vis, x, y, scale_x, scale_y) for vis, x, y in item["labels"]],
            dim=0,
        )
        return inputs, heatmaps

    def get_info(self, idx):
        """Return metadata dict for sample *idx*."""
        return self.data_items[idx]


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "datasets/trackNet"

    dataset = TrackNetV1CSVDataset(root)
    print(f"Dataset size: {len(dataset)}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for inputs, heatmaps in loader:
        print(f"inputs:   {inputs.shape}  range [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"heatmaps: {heatmaps.shape}  range [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")
        break
