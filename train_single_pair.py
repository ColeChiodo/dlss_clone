import torch
import torch.nn as nn
import torch.optim as optim
from models.sr_model import SuperResolutionNet
from utils.data_utils import load_qrd_pair
from torchvision.utils import save_image
import os

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SuperResolutionNet(scale_factor=2).to(device)

    # Load one low/high-res pair (edit paths as needed)
    lr, hr = load_qrd_pair('qrd_data/360p/MipBiasMinus2/0000/0000.png', 
                           'qrd_data/1080p/Native/0000/0000.png', device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("output", exist_ok=True)

    for epoch in range(200):
        model.train()
        sr = model(lr)
        loss = criterion(sr, hr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
            save_image(sr.clamp(0, 1), f"output/sr_epoch_{epoch+1}.png")

    torch.save(model.state_dict(), "output/sr_single_image_model.pth")

if __name__ == "__main__":
    train()
    print("Training complete.")
