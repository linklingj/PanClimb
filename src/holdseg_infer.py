
import argparse
import json
from random import randint
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

from holdseg import get_maskrcnn_model

parser = argparse.ArgumentParser(description="HoldSeg Inference")
parser.add_argument("-m", required=True, help="Path to the trained model file.")
parser.add_argument("-i", required=True, help="Path to the input image.")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

SCORE_THRESH = 0.3
CATEGORY_ID = 1
NUM_CLASSES = 2
OUTPUT_JSON = "outputs/holdseg_predictions.json"

@torch.no_grad()
def main():
    args = parser.parse_args()
    model_path = args.m
    img_path = args.i
    model = get_maskrcnn_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]
    if isinstance(checkpoint, dict) and any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        print(f"[WARN] Missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        print(f"[WARN] Unexpected keys when loading checkpoint: {len(unexpected)}")
    model.to(device=device)
    model.eval()

    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img_tensor = img_tensor.to(device=device)

    output = model(img_tensor)[0]

    boxes = output["boxes"].cpu()
    scores = output["scores"].cpu()
    masks = output["masks"].cpu()

    predictions = []

    for i in range(len(scores)):
        score = float(scores[i])
        if score < SCORE_THRESH:
            continue

        x1, y1, x2, y2 = boxes[i].tolist()
        bbox = [x1, y1, x2 - x1, y2 - y1]  # Convert to [x, y, w, h] format
        mask = masks[i, 0].numpy() > 0.5
        mask = mask.astype(np.uint8)

        # pycocotools expects a Fortran-contiguous array of shape (H, W, N)
        mask_f = np.asfortranarray(mask[:, :, None])
        rle = coco_mask.encode(mask_f)[0]
        rle["counts"] = rle["counts"].decode("utf-8")
        
        predictions.append({
            "id": i,
            "image_id": 1,
            "category_id": CATEGORY_ID,
            "segmentation": rle,
            "bbox": bbox,
            "area": int(mask.sum()),
            "score": score,
            "iscrowd": 0
        })

    content = {"images": [{"id": 1, "width": w, "height": h, "file_name": Path(img_path).name}],
               "annotations": predictions,
               "categories": [{"id": CATEGORY_ID, "name": "hold"}]}
    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(content, f)

    print(f"Saved {len(predictions)} predictions to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
