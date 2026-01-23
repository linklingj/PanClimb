
import json
from typing import Any, Dict
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset

import torchvision

# coco
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

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
        self.img_ids = sorted(self.coco.getImgIds())
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.img_ids)
    
    def load_image(self, idx: int):
        pass

    def ann_to_mask(self, ann:Dict[str, Any], height: int, width: int) -> torch.Tensor:
        pass
    
    def __getitem__(self, idx: int):          
        pass

def get_maskrcnn_model(num_classes: int):
    weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    return model

def main():
    num_classes = 2
    batch_size = 4
    lr = 0.005
    epochs = 10

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_dataset = Coco

if __name__ == "__main__":
    main()