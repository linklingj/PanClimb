
import json, os
from itertools import cycle
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm
from PIL import Image

import torchvision

# coco
import pycocotools
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from utils.cocoviewer import coco_viewer

# ------
# Dataset: COCO json -> PyTorch (image, target)
# target = {
#     "boxes":   FloatTensor[N, 4],   # x1,y1,x2,y2
#     "labels":  Int64Tensor[N],      # 클래스 id
#     "masks":   UInt8Tensor[N,H,W],  # binary mask
#     "image_id": Int64Tensor[1],
#     "area":    FloatTensor[N],
#     "iscrowd": Int64Tensor[N],
# }
# ------
class CocoSegDataset(Dataset):
    def __init__(self,
                 img_dir : str,
                 ann_file : str,
                 transforms=None):
        super().__init__()

        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        # keep only images that actually have annotations/boxes
        all_img_ids = sorted(self.coco.getImgIds())
        self.img_ids = [
            img_id for img_id in all_img_ids
            if len(self.coco.getAnnIds(imgIds=img_id)) > 0
        ]
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_ids)
    
    def load_image(self, img_info: Dict[str, Any]) -> Image.Image:
        img_path = os.path.join(self.img_dir, img_info["file_name"])
        img = Image.open(img_path).convert("RGB")
        return img

    # Convert COCO annotation to binary mask (H, W, 1)
    def ann_to_mask(self, ann:Dict[str, Any], height: int, width: int) -> torch.Tensor:
        seg = ann.get("segmentation", None)
        if seg is None:
            raise ValueError("Annotation does not contain 'segmentation' field.")
        elif isinstance(seg, list):
            rles = coco_mask.frPyObjects(seg, height, width)
            rle = coco_mask.merge(rles)
            m = coco_mask.decode(rle)
        elif isinstance(seg, dict):
            m = coco_mask.decode(seg)
        else:
            raise TypeError("Segmentation format not supported.")

        m = (m>0).astype("uint8")
        return torch.from_numpy(m)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:          
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img = self.load_image(img_info)

        width, height = img.size
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        masks = []
        labels = []
        areas = []
        iscrowd = []

        for ann in anns:
            # [x, y, w, h]
            bbox = ann["bbox"]
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = x1 + bbox[2]
            y2 = y1 + bbox[3]
            boxes.append([x1, y1, x2, y2])

            labels.append(1) # ann["category_id"]) = 1

            masks.append(self.ann_to_mask(ann, height, width))

            areas.append(ann["area"])

            iscrowd.append(ann.get("iscrowd", 0))

        if len(boxes) == 0:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            masks_t = torch.zeros((0, height, width), dtype=torch.uint8)
            areas_t = torch.zeros((0,), dtype=torch.float32)
            iscrowd_t = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            masks_t = torch.stack(masks, dim=0).to(torch.uint8)
            areas_t = torch.tensor(areas, dtype=torch.float32)
            iscrowd_t = torch.tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "masks": masks_t,
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": areas_t,
            "iscrowd": iscrowd_t
        }

        if not torch.is_tensor(img):
            img = torchvision.transforms.functional.to_tensor(img)

        return img, target


def get_maskrcnn_model(num_classes: int):
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features_box, num_classes
    )

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq: int = 50):
    model.train()
    
    loss = 0.0
    for step, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        print(f"[DEBUG]: {[t['image_id'] for t in targets]}")


        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        loss += loss_value

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if (step + 1) % print_freq == 0:
            print(f"Epoch [{epoch+1}], Step [{step+1}/{len(data_loader)}], Loss: {loss_value:.4f}")

    return loss / max(1,len(data_loader))

@torch.no_grad()
def evaluate(model, data_loader, device, score_thresh: float = 0.5):
    model.eval()

    total_preds = 0
    total_imgs = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for out in outputs:
            score = out["scores"].cpu()
            keep = score >= score_thresh
            total_preds += keep.sum().item()
            total_imgs += 1

    avg_preds = total_preds / max(1, total_imgs)
    print(f"[EVAL] Average Predictions per Image (score >= {score_thresh}): {avg_preds:.2f}")

def main():
    num_classes = 2
    batch_size = 4
    lr = 0.005
    weight_decay = 0.0005
    epochs = 1

    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT_DIR / "data"
    TRAIN_IMG_PATH = DATA_DIR / "images/bh"
    TRAIN_ANN_PATH = DATA_DIR / "annotation/bh-coco.json"
    TEST_IMG_PATH = DATA_DIR / "images/bh-phone"
    TEST_ANN_PATH = DATA_DIR / "annotation/bh-phone-coco.json"
    MODEL_SAVE_PATH = ROOT_DIR / "models/checkpoints"

    # Show Train Dataset
    coco_viewer(str(TRAIN_ANN_PATH), str(TRAIN_IMG_PATH))
    
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        raise EnvironmentError("CUDA or MPS device not found. A GPU is required to run this code.")
    print(f"Using device: {device}")

    train_dataset = CocoSegDataset(
        img_dir=TRAIN_IMG_PATH,
        ann_file=TRAIN_ANN_PATH,
        transforms=None
    )
    test_dataset = CocoSegDataset(
        img_dir=TEST_IMG_PATH,
        ann_file=TEST_ANN_PATH,
        transforms=None
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    model_0 = get_maskrcnn_model(num_classes=num_classes)
    model_0.to(device)

    params = [p for p in model_0.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr,
                                momentum=0.9, weight_decay=weight_decay)
    
    for epoch in tqdm(range(epochs)):
        avg_loss = train_one_epoch(model_0, optimizer, train_loader, device, epoch, print_freq=50)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        evaluate(model_0, test_loader, device, score_thresh=0.5)

        torch.save(model_0.state_dict(), MODEL_SAVE_PATH / f"maskrcnn_epoch{epoch+1}.pth")

    print("Training complete.")
    


if __name__ == "__main__":
    main()
