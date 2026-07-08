"""
TrackNet V1 Dataset Loader

Reads the yastrebksv/TrackNet dataset directly without preprocessing.

Dataset Structure (https://github.com/yastrebksv/TrackNet):
    datasets/trackNet/
    ├── images/
    │   ├── game1/
    │   │   ├── Clip1/
    │   │   │   ├── 0000.jpg
    │   │   │   └── ...
    │   │   └── Clip13/
    │   └── game10/
    └── gts/
        ├── game1/
        │   ├── Clip1/
        │   │   ├── 0000.jpg
        │   │   └── ...
        │   └── Clip13/
        └── game10/

Output Format:
    - inputs:  (9, 288, 512) — 3 RGB frames concatenated, normalized to [0, 1]
    - heatmaps: (3, 288, 512) — 3 grayscale heatmaps concatenated, normalized to [0, 1]
"""

import glob
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

TARGET_WIDTH = 512
TARGET_HEIGHT = 288


class TrackNetV1Dataset(Dataset):
    """PyTorch Dataset for the yastrebksv/TrackNet tennis dataset.

    Reads frames from ``images/game*/Clip*/`` and the corresponding ground-truth
    heatmaps from ``gts/game*/Clip*/``.  Frames are resized on-the-fly from the
    original 1280×720 to 512×288.  Ground-truth images are RGB with equal R/G/B
    values; they are converted to grayscale so the output matches the format
    expected by TrackNet-v4.
    """

    def __init__(self, root_dir):
        """
        Args:
            root_dir: Root directory of the dataset (must contain ``images/`` and
                      ``gts/`` subdirectories).
        """
        self.root_dir = Path(root_dir)
        self.transform = transforms.Compose([
            transforms.Resize((TARGET_HEIGHT, TARGET_WIDTH)),
            transforms.ToTensor(),
        ])
        self.heatmap_transform = transforms.Compose([
            transforms.Resize((TARGET_HEIGHT, TARGET_WIDTH)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        self.data_items = self._scan_dataset()

    def _scan_dataset(self):
        images_root = self.root_dir / "images"
        gts_root = self.root_dir / "gts"

        if not images_root.exists():
            # Fallback: game directories may live directly under root (no images/ wrapper)
            non_gts_dirs = [d for d in self.root_dir.iterdir() if d.is_dir() and d.name != "gts"]
            if non_gts_dirs:
                images_root = self.root_dir
            else:
                raise FileNotFoundError(
                    f"images/ directory not found under {self.root_dir}. "
                    f"Expected either:\n"
                    f"  {self.root_dir}/images/game*/Clip*/*.jpg  (with gts/ sibling)\n"
                    f"  {self.root_dir}/game*/Clip*/*.jpg          (game dirs at root, with gts/ sibling)"
                )
        if not gts_root.exists():
            raise FileNotFoundError(
                f"gts/ directory not found under {self.root_dir}. "
                f"Expected: {self.root_dir}/gts/game*/Clip*/*.jpg"
            )

        items = []
        game_dirs = sorted(d for d in images_root.iterdir() if d.is_dir())
        print(f"Scanning {len(game_dirs)} game folders...")

        for game_dir in game_dirs:
            clip_dirs = sorted(d for d in game_dir.iterdir() if d.is_dir())
            for clip_dir in clip_dirs:
                gt_clip_dir = gts_root / game_dir.name / clip_dir.name
                if not gt_clip_dir.exists():
                    continue
                items.extend(self._process_clip(clip_dir, gt_clip_dir))

        print(f"Found {len(items)} valid samples")
        return items

    def _process_clip(self, img_clip_dir, gt_clip_dir):
        img_files = self._get_sorted_images(img_clip_dir)
        gt_files = self._get_sorted_images(gt_clip_dir)

        # Build a lookup from stem to path for ground-truth files
        gt_lookup = {Path(p).stem: p for p in gt_files}

        # Keep only frames that have a matching ground-truth
        paired = [(p, gt_lookup[Path(p).stem]) for p in img_files if Path(p).stem in gt_lookup]

        if len(paired) < 3:
            return []

        img_paired, gt_paired = zip(*paired)

        return [
            {
                "inputs": list(img_paired[i:i + 3]),
                "heatmaps": list(gt_paired[i:i + 3]),
                "game": img_clip_dir.parent.name,
                "clip": img_clip_dir.name,
                "idx": i,
            }
            for i in range(len(img_paired) - 2)
        ]

    @staticmethod
    def _get_sorted_images(directory):
        """Return image paths sorted by numeric stem (e.g. 0000 < 0001)."""
        files = glob.glob(str(directory / "*.jpg"))
        return sorted(files, key=lambda x: int(Path(x).stem))

    def _load_image(self, image_path, is_heatmap=False):
        try:
            image = Image.open(image_path).convert("RGB")
            if is_heatmap:
                return self.heatmap_transform(image)
            return self.transform(image)
        except Exception:
            channels = 1 if is_heatmap else 3
            return torch.zeros(channels, TARGET_HEIGHT, TARGET_WIDTH)

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        """
        Returns:
            inputs:   (9, 288, 512) — 3 RGB frames concatenated, values in [0, 1]
            heatmaps: (3, 288, 512) — 3 grayscale heatmaps concatenated, values in [0, 1]
        """
        item = self.data_items[idx]
        inputs = torch.cat([self._load_image(p, False) for p in item["inputs"]], dim=0)
        heatmaps = torch.cat([self._load_image(p, True) for p in item["heatmaps"]], dim=0)
        return inputs, heatmaps

    def get_info(self, idx):
        """Return metadata for a sample."""
        return self.data_items[idx]


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "datasets/trackNet"

    dataset = TrackNetV1Dataset(root)
    print(f"Dataset size: {len(dataset)}")

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
    for inputs, heatmaps in loader:
        print(f"inputs:   {inputs.shape}  range [{inputs.min():.3f}, {inputs.max():.3f}]")
        print(f"heatmaps: {heatmaps.shape}  range [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")
        break
