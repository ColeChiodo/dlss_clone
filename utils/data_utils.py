from PIL import Image
import torchvision.transforms as T
import torch

def load_qrd_pair(low_path, high_path, device='cpu'):
    transform = T.ToTensor()

    lr = Image.open(low_path).convert("RGB")
    hr = Image.open(high_path).convert("RGB")

    lr_tensor = transform(lr).unsqueeze(0).to(device)
    hr_tensor = transform(hr).unsqueeze(0).to(device)

    return lr_tensor, hr_tensor

def preprocess_image(img_path, device='cpu'):
    img = Image.open(img_path).convert('RGB')
    transform = T.ToTensor()
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor
