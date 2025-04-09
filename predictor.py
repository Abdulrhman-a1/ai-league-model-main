import torch
from model import CSRNet
from torchvision import transforms
from PIL import Image
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CSRNet().to(device)
model.load_state_dict(torch.load("csrnet_weights.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((384, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def predict_density(image: Image.Image):
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        count = output.sum().item()
        density_map = output.squeeze().cpu().numpy()
    return int(count), density_map
