# CROP
# python src/holdtype_ssl.py build-crops \
#   --image data/sample/theclimb1.jpeg \
#   --seg-json outputs/holdseg_predictions.json \
#   --output-dir data/holdtype/crops \
#   --metadata-out data/holdtype/crops_metadata.csv




from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

HOLD_TYPES = [
    "Jug",
    "Sloper",
    "Crimp",
    "Pinch",
    "Pocket",
    "FootHold",
    "Sidepull",
    "Undercling",
    "Volume",
    "DownHold"
]
HOLD_TYPE_TO_ID = {name: idx for idx, name in enumerate(HOLD_TYPES)}

# Global configs (non-essential args moved from CLI)
DEFAULT_CROP_PADDING = 12
DEFAULT_CROP_MIN_SIZE = 16
PRETRAIN_EPOCHS = 30
PRETRAIN_BATCH_SIZE = 64
PRETRAIN_LR = 3e-4
PRETRAIN_TEMPERATURE = 0.2
PRETRAIN_NUM_WORKERS = 0
FINETUNE_EPOCHS = 30
FINETUNE_BATCH_SIZE = 32
FINETUNE_LR = 1e-4
FINETUNE_NUM_WORKERS = 0
FINETUNE_SSL_ENCODER_PATH: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hold type SSL pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    crop = subparsers.add_parser("build-crops", help="Build masked hold crops from holdseg json")
    crop.add_argument("--image", required=True, help="Source image path")
    crop.add_argument("--seg-json", required=True, help="HoldSeg prediction json path")
    crop.add_argument("--output-dir", required=True, help="Output crop directory")
    crop.add_argument("--metadata-out", required=True, help="Output crop metadata csv")

    pretrain = subparsers.add_parser("pretrain", help="SimCLR pretrain with unlabeled crops")
    pretrain.add_argument("--metadata", required=True, help="Crop metadata csv")
    pretrain.add_argument("--save-path", required=True, help="SSL encoder checkpoint path")

    finetune = subparsers.add_parser("finetune", help="Supervised finetune using labeled crops")
    finetune.add_argument("--metadata", required=True, help="Labeled crop metadata csv with label column")
    finetune.add_argument("--save-path", required=True, help="Classifier checkpoint path")

    return parser.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def polygon_to_mask(height: int, width: int, segmentation: List[List[float]]) -> np.ndarray:
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)
    for poly in segmentation:
        if len(poly) < 6:
            continue
        pts = [(poly[idx], poly[idx + 1]) for idx in range(0, len(poly), 2)]
        draw.polygon(pts, outline=1, fill=1)
    return np.array(mask_img, dtype=np.uint8)


def crop_masked_hold(
    image: Image.Image,
    ann: Dict,
    image_w: int,
    image_h: int,
    padding: int,
) -> Optional[Image.Image]:
    bbox = ann["bbox"]
    x, y, w, h = bbox
    x1 = max(0, int(np.floor(x - padding)))
    y1 = max(0, int(np.floor(y - padding)))
    x2 = min(image_w, int(np.ceil(x + w + padding)))
    y2 = min(image_h, int(np.ceil(y + h + padding)))
    if x2 <= x1 or y2 <= y1:
        return None

    crop_rgb = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))
    seg = ann.get("segmentation", [])
    if not seg:
        return Image.fromarray(crop_rgb)

    full_mask = polygon_to_mask(image_h, image_w, seg)
    crop_mask = full_mask[y1:y2, x1:x2][:, :, None]
    crop_rgb = crop_rgb * crop_mask
    return Image.fromarray(crop_rgb.astype(np.uint8))


def build_crops(
    image_path: str,
    seg_json_path: str,
    output_dir: str,
    metadata_out: str,
    padding: int,
    min_size: int,
) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "crop_path",
        "image_path",
        "seg_json_path",
        "ann_id",
        "category_id",
        "bbox_x",
        "bbox_y",
        "bbox_w",
        "bbox_h",
    ]
    written = 0
    skipped = 0

    with open(metadata_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        img_path = Path(image_path)
        seg_path = Path(seg_json_path)
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image: {img_path}")
        if not seg_path.exists():
            raise FileNotFoundError(f"Missing segmentation json: {seg_path}")

        image = Image.open(img_path).convert("RGB")
        image_w, image_h = image.size
        seg_json = json.loads(seg_path.read_text(encoding="utf-8"))
        anns = seg_json.get("annotations", [])

        for ann in anns:
            ann_id = int(ann["id"])
            hold_img = crop_masked_hold(image, ann, image_w, image_h, padding=padding)
            if hold_img is None:
                skipped += 1
                continue
            if hold_img.width < min_size or hold_img.height < min_size:
                skipped += 1
                continue

            crop_name = f"{img_path.stem}_ann{ann_id:05d}.png"
            crop_path = out_dir / crop_name
            hold_img.save(crop_path)

            writer.writerow(
                {
                    "crop_path": str(crop_path),
                    "image_path": str(img_path),
                    "seg_json_path": str(seg_path),
                    "ann_id": ann_id,
                    "category_id": int(ann.get("category_id", 1)),
                    "bbox_x": ann["bbox"][0],
                    "bbox_y": ann["bbox"][1],
                    "bbox_w": ann["bbox"][2],
                    "bbox_h": ann["bbox"][3],
                }
            )
            written += 1

    print(f"[DONE] Crops written: {written}, skipped: {skipped}, metadata: {metadata_out}")


@dataclass
class Sample:
    crop_path: str
    label: Optional[int]


def read_crop_metadata(path: str, require_label: bool) -> List[Sample]:
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "crop_path" not in reader.fieldnames:
            raise ValueError("metadata must include crop_path column")
        if require_label and "label" not in reader.fieldnames:
            raise ValueError("labeled metadata must include label column")
        samples: List[Sample] = []
        for row in reader:
            label_name = row.get("label", "").strip()
            if require_label:
                if label_name not in HOLD_TYPE_TO_ID:
                    continue
                label = HOLD_TYPE_TO_ID[label_name]
            else:
                label = HOLD_TYPE_TO_ID[label_name] if label_name in HOLD_TYPE_TO_ID else None
            samples.append(Sample(crop_path=row["crop_path"], label=label))
    return samples


class SSLHoldDataset(Dataset):
    def __init__(self, samples: List[Sample], image_size: int = 224):
        self.samples = samples
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                transforms.RandomGrayscale(p=0.15),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = Image.open(self.samples[idx].crop_path).convert("RGB")
        v1 = self.transform(img)
        v2 = self.transform(img)
        return v1, v2


class LabeledHoldDataset(Dataset):
    def __init__(self, samples: List[Sample], image_size: int = 224):
        self.samples = [s for s in samples if s.label is not None]
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        img = Image.open(sample.crop_path).convert("RGB")
        x = self.transform(img)
        y = torch.tensor(sample.label, dtype=torch.long)
        return x, y


class SimCLRModel(nn.Module):
    def __init__(self, proj_dim: int = 128):
        super().__init__()
        encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return h, z


class HoldClassifier(nn.Module):
    def __init__(self, num_classes: int = len(HOLD_TYPES)):
        super().__init__()
        encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.classifier(h)


def simclr_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    logits = torch.mm(z, z.t()) / temperature
    mask = torch.eye(2 * batch_size, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, -1e9)

    positives = torch.cat(
        [
            torch.diag(logits, batch_size),
            torch.diag(logits, -batch_size),
        ],
        dim=0,
    )
    denominator = torch.logsumexp(logits, dim=1)
    loss = -(positives - denominator).mean()
    return loss


def run_pretrain(args: argparse.Namespace) -> None:
    device = get_device()
    samples = read_crop_metadata(args.metadata, require_label=False)
    dataset = SSLHoldDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=PRETRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=PRETRAIN_NUM_WORKERS,
        drop_last=True,
    )
    model = SimCLRModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=PRETRAIN_LR)

    print(f"[INFO] Device: {device}, pretrain samples: {len(dataset)}")
    print(
        f"[INFO] pretrain config: epochs={PRETRAIN_EPOCHS}, batch_size={PRETRAIN_BATCH_SIZE}, "
        f"lr={PRETRAIN_LR}, temperature={PRETRAIN_TEMPERATURE}"
    )
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)
            _, z1 = model(v1)
            _, z2 = model(v2)
            loss = simclr_loss(z1, z2, temperature=PRETRAIN_TEMPERATURE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / max(1, len(loader))
        print(f"[PRETRAIN] Epoch {epoch + 1}/{PRETRAIN_EPOCHS} loss={avg_loss:.4f}")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": model.encoder.state_dict(), "hold_types": HOLD_TYPES}, save_path)
    print(f"[DONE] Saved SSL encoder to {save_path}")


def run_finetune(args: argparse.Namespace) -> None:
    device = get_device()
    samples = read_crop_metadata(args.metadata, require_label=True)
    dataset = LabeledHoldDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=FINETUNE_BATCH_SIZE,
        shuffle=True,
        num_workers=FINETUNE_NUM_WORKERS,
    )
    if len(dataset) == 0:
        raise ValueError("No labeled samples found in metadata")

    model = HoldClassifier(num_classes=len(HOLD_TYPES)).to(device)
    if FINETUNE_SSL_ENCODER_PATH:
        checkpoint = torch.load(FINETUNE_SSL_ENCODER_PATH, map_location=device)
        encoder_state = checkpoint.get("encoder", checkpoint)
        model.encoder.load_state_dict(encoder_state, strict=False)
        print(f"[INFO] Loaded SSL encoder from {FINETUNE_SSL_ENCODER_PATH}")

    opt = torch.optim.Adam(model.parameters(), lr=FINETUNE_LR)
    criterion = nn.CrossEntropyLoss()

    print(f"[INFO] Device: {device}, finetune samples: {len(dataset)}")
    print(
        f"[INFO] finetune config: epochs={FINETUNE_EPOCHS}, batch_size={FINETUNE_BATCH_SIZE}, "
        f"lr={FINETUNE_LR}, ssl_encoder={FINETUNE_SSL_ENCODER_PATH}"
    )
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
        avg_loss = epoch_loss / max(1, len(loader))
        acc = 100.0 * correct / max(1, total)
        print(f"[FINETUNE] Epoch {epoch + 1}/{FINETUNE_EPOCHS} loss={avg_loss:.4f} acc={acc:.2f}%")

    save_path = Path(args.save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hold_types": HOLD_TYPES,
            "input_size": 224,
        },
        save_path,
    )
    print(f"[DONE] Saved classifier checkpoint to {save_path}")


def main() -> None:
    args = parse_args()
    if args.command == "build-crops":
        build_crops(
            image_path=args.image,
            seg_json_path=args.seg_json,
            output_dir=args.output_dir,
            metadata_out=args.metadata_out,
            padding=DEFAULT_CROP_PADDING,
            min_size=DEFAULT_CROP_MIN_SIZE,
        )
        return
    if args.command == "pretrain":
        run_pretrain(args)
        return
    if args.command == "finetune":
        run_finetune(args)
        return
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
