import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms as T
from models.sr_model import SuperResolutionNet
from dataset.qrd_dataset import QRDDataset
from torchvision.utils import save_image
import os

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scale_factor = 2
    crop_size = 128

    print(f"Training on {device} with scale factor {scale_factor} and crop size {crop_size}")

    model = SuperResolutionNet(scale_factor=scale_factor).to(device)

    # Transforms
    transform_lr = T.Compose([
        T.CenterCrop(crop_size // scale_factor),
        T.ToTensor()
    ])

    transform_hr = T.Compose([
        T.CenterCrop(crop_size),
        T.ToTensor()
    ])

    dataset = QRDDataset(
        base_dir='qrd_data/',
        transform_lr=transform_lr,
        transform_hr=transform_hr
    )

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    os.makedirs("output", exist_ok=True)

    for epoch in range(100):  # Changed to 100 epochs
        model.train()
        total_loss = 0

        for i, (lr, hr) in enumerate(dataloader):
            lr, hr = lr.to(device), hr.to(device)
            sr = model(lr)
            loss = criterion(sr, hr)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

        # Save sample output
        with torch.no_grad():
            save_image(sr, f"output/sample_epoch_{epoch+1}.png")

        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"output/sr_model_epoch_{epoch+1}.pth")
            print(f"Model saved at epoch {epoch+1}")

    # Save final model after training completes
    torch.save(model.state_dict(), "output/sr_full_dataset_model_final.pth")
    print("Training complete.")

if __name__ == "__main__":
    train()
