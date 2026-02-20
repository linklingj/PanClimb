from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import models, transforms

DEFAULT_HOLD_TYPES = [
    "Jug",
    "Sloper",
    "Crimp",
    "Pinch",
    "Pocket",
    "FootHold",
    "Sidepull",
    "Undercling",
    "Volume",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hold type inference on holdseg results")
    parser.add_argument("--model", required=True, help="Path to holdtype classifier checkpoint")
    parser.add_argument("--image", required=True, help="Path to source image")
    parser.add_argument("--holdseg-json", required=True, help="Path to holdseg prediction json")
    parser.add_argument("--output", required=True, help="Output json path")
    parser.add_argument("--padding", type=int, default=12)
    parser.add_argument("--override-volume-from-seg", action="store_true")
    return parser.parse_args()


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class HoldClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = encoder.fc.in_features
        encoder.fc = nn.Identity()
        self.encoder = encoder
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.classifier(h)


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
    x, y, w, h = ann["bbox"]
    x1 = max(0, int(np.floor(x - padding)))
    y1 = max(0, int(np.floor(y - padding)))
    x2 = min(image_w, int(np.ceil(x + w + padding)))
    y2 = min(image_h, int(np.ceil(y + h + padding)))
    if x2 <= x1 or y2 <= y1:
        return None

    crop_rgb = np.array(image.crop((x1, y1, x2, y2)).convert("RGB"))
    seg = ann.get("segmentation", [])
    if seg:
        full_mask = polygon_to_mask(image_h, image_w, seg)
        crop_mask = full_mask[y1:y2, x1:x2][:, :, None]
        crop_rgb = crop_rgb * crop_mask
    return Image.fromarray(crop_rgb.astype(np.uint8))


def main() -> None:
    args = parse_args()
    device = get_device()

    checkpoint = torch.load(args.model, map_location=device)
    hold_types = checkpoint.get("hold_types", DEFAULT_HOLD_TYPES)
    input_size = int(checkpoint.get("input_size", 224))
    model = HoldClassifier(num_classes=len(hold_types))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    tfm = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
        ]
    )

    image = Image.open(args.image).convert("RGB")
    image_w, image_h = image.size
    holdseg = json.loads(Path(args.holdseg_json).read_text(encoding="utf-8"))
    annotations = holdseg.get("annotations", [])

    with torch.no_grad():
        for ann in annotations:
            if args.override_volume_from_seg and int(ann.get("category_id", 1)) == 2:
                ann["hold_type"] = "Volume"
                ann["hold_type_id"] = hold_types.index("Volume") if "Volume" in hold_types else -1
                ann["hold_type_score"] = 1.0
                continue

            crop = crop_masked_hold(image, ann, image_w, image_h, padding=args.padding)
            if crop is None:
                ann["hold_type"] = "Unknown"
                ann["hold_type_id"] = -1
                ann["hold_type_score"] = 0.0
                continue

            x = tfm(crop).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]
            hold_type_id = int(torch.argmax(probs).item())
            ann["hold_type"] = hold_types[hold_type_id]
            ann["hold_type_id"] = hold_type_id
            ann["hold_type_score"] = float(probs[hold_type_id].item())

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(holdseg, indent=2), encoding="utf-8")
    print(f"[DONE] Saved hold type inference to {out_path}")


if __name__ == "__main__":
    main()
