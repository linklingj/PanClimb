
"""
Run Mask R-CNN inference and export COCO-style polygons for holds.
SAMPLE: python .\src\holdseg_infer.py -m .\models\checkpoints\maskrcnn_epoch3.pth -i .\data\sample\ theclimb1.jpeg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from holdseg import get_maskrcnn_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCORE_THRESH = 0.3
NUM_CLASSES = 3
CATEGORY = [{"id": 1, "name": "hold"}, {"id": 2, "name": "volume"}]
DEFAULT_OUTPUT = "outputs/holdseg_predictions.json"


# ---------- Geometry helpers ----------
def mask_to_polygon(mask: np.ndarray) -> List[List[float]]:
    """
    Convert a binary mask (H, W) to a COCO polygon: [[x1, y1, x2, y2, ...]].
    Uses a monotone chain convex hull to avoid extra dependencies (cv2/skimage).
    """
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []

    points = np.stack([xs, ys], axis=1)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = np.concatenate((lower[:-1], upper[:-1]))

    if len(hull) < 3:  # fallback to box if hull is too small
        x_min, y_min = xs.min(), ys.min()
        x_max, y_max = xs.max(), ys.max()
        hull = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])

    return [hull.astype(float).reshape(-1).tolist()]


# ---------- IO helpers ----------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HoldSeg inference")
    parser.add_argument("-m", "--model", required=True, help="Path to trained model (.pth).")
    parser.add_argument("-i", "--image", required=True, help="Path to input image.")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Where to save COCO json.")
    parser.add_argument("--score-thresh", type=float, default=SCORE_THRESH,
                        help=f"Confidence threshold (default: {SCORE_THRESH}).")
    return parser.parse_args()


def load_checkpoint(model_path: str, device: str) -> Dict:
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict):
        checkpoint = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        if any(k.startswith("module.") for k in checkpoint):
            checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    return checkpoint


def build_model(model_path: str, device: str) -> torch.nn.Module:
    model = get_maskrcnn_model(num_classes=NUM_CLASSES)
    model.roi_heads.detections_per_img = 300
    model.roi_heads.score_thresh = 0.0
    model.roi_heads.nms_thresh = 0.5
    checkpoint = load_checkpoint(model_path, device)
    model.load_state_dict(checkpoint, strict=False)

    model.to(device=device)
    model.eval()
    return model


def load_image(img_path: str) -> Tuple[Image.Image, torch.Tensor]:
    img = Image.open(img_path).convert("RGB")
    tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img, tensor


def run_inference(model: torch.nn.Module, img_tensor: torch.Tensor, score_thresh: float) -> Dict[str, torch.Tensor]:
    output = model(img_tensor.to(DEVICE))[0]
    keep = output["scores"] >= score_thresh
    return {
        "category_ids": output["labels"][keep].cpu(),
        "boxes": output["boxes"][keep].cpu(),
        "scores": output["scores"][keep].cpu(),
        "masks": output["masks"][keep].cpu(),
    }


def build_annotations(det: Dict[str, torch.Tensor], image_id: int) -> List[Dict]:
    category_ids, boxes, scores, masks = det["category_ids"], det["boxes"], det["scores"], det["masks"]
    anns = []

    for idx, (category_id, box, score, mask_t) in enumerate(zip(category_ids, boxes, scores, masks)):
        x1, y1, x2, y2 = box.tolist()
        bbox = [x1, y1, x2 - x1, y2 - y1]  # [x, y, w, h]
        mask = (mask_t[0].numpy() > 0.5).astype(np.uint8)

        anns.append({
            "id": idx,
            "image_id": image_id,
            "category_id": category_id.item(),
            "segmentation": mask_to_polygon(mask),
            "bbox": bbox,
            "area": int(mask.sum()),
            "score": float(score),
            "iscrowd": 0,
        })
    return anns


@torch.no_grad()
def main():
    args = parse_args()

    print(f"[INFO] Device: {DEVICE}")
    model = build_model(args.model, DEVICE)

    img, img_tensor = load_image(args.image)
    width, height = img.size

    detections = run_inference(model, img_tensor, args.score_thresh)
    annotations = build_annotations(detections, image_id=1)

    output = {
        "images": [{"id": 1, "width": width, "height": height, "file_name": Path(args.image).name}],
        "annotations": annotations,
        "categories": CATEGORY,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[DONE] Saved {len(annotations)} predictions to {args.output}")


if __name__ == "__main__":
    main()
