import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class QRDDataset(Dataset):
    def __init__(self, base_dir, low_res='360p', high_res='1080p',
                 low_type=None, high_type='Native',
                 transform_lr=None, transform_hr=None):
        self.lr_base = os.path.join(base_dir, low_res)
        self.hr_base = os.path.join(base_dir, high_res)

        if low_type is None:
            low_type = self._find_mipbias_folder(self.lr_base)

        self.lr_base = os.path.join(self.lr_base, low_type)
        self.hr_base = os.path.join(self.hr_base, high_type)

        self.transform_lr = transform_lr or T.ToTensor()
        self.transform_hr = transform_hr or T.ToTensor()

        self.pairs = []
        scenes = sorted(os.listdir(self.lr_base))
        for scene in scenes:
            lr_scene_dir = os.path.join(self.lr_base, scene)
            hr_scene_dir = os.path.join(self.hr_base, scene)
            if not os.path.isdir(lr_scene_dir) or not os.path.isdir(hr_scene_dir):
                continue
            frame_files = sorted(os.listdir(lr_scene_dir))
            for frame_file in frame_files:
                if frame_file.endswith('.png'):
                    self.pairs.append({
                        "lr_path": os.path.join(lr_scene_dir, frame_file),
                        "hr_path": os.path.join(hr_scene_dir, frame_file)
                    })

    def _find_mipbias_folder(self, path):
        for name in os.listdir(path):
            if name.lower().startswith("mipbias"):
                return name
        raise FileNotFoundError(f"No MipBias* folder found in {path}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        lr_img = Image.open(pair["lr_path"]).convert("RGB")
        hr_img = Image.open(pair["hr_path"]).convert("RGB")
        return self.transform_lr(lr_img), self.transform_hr(hr_img)
