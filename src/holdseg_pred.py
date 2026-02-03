import torch
import torchvision
from PIL import Image

def load_model(model_path: str) -> torch.nn.Module:
    model = torch.load(model_path)
    return model

def load_image(image_path: str) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    transform = torch.nn.Sequential(
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
    )
    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

def holdseg_predict(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        predictions = model(images)
    return predictions

model = load_model("models/checkpoints/maskrcnn_epoch0127.pth")
img = load_image("data/sample/theclimb1.jpeg")
pred = holdseg_predict(model, img)

print(pred)