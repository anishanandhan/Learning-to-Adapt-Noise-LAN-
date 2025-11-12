import torch
from pathlib import Path
from torchvision.io import read_image

class Dataset(torch.utils.data.Dataset):
    def __init__(self, lq_dir, gt_dir, crop_size=256):
        self.lq_dir = Path(lq_dir)
        self.gt_dir = Path(gt_dir)
        self.crop_size = crop_size
        self.lq_paths = sorted(list(self.lq_dir.glob("*.png")))
        self.gt_paths = sorted(list(self.gt_dir.glob("*.png")))
        assert len(self.lq_paths) == len(self.gt_paths)

    def __len__(self):
        return len(self.lq_paths)

    def __getitem__(self, idx):
        lq_path = self.lq_paths[idx]
        gt_path = self.gt_paths[idx]
        lq_name = self.lq_paths[idx].stem 

        lq = read_image(str(lq_path)) / 255.0
        gt = read_image(str(gt_path)) / 255.0
        
        # --- MODIFICATION: Re-enabled the cropping logic ---
        H, W = lq.shape[1:]
        i, j = torch.randint(H - self.crop_size, (1,))[0], torch.randint(W - self.crop_size, (1,))[0]
        lq_tensor = lq[:, i:i+self.crop_size, j:j+self.crop_size]
        gt_tensor = gt[:, i:i+self.crop_size, j:j+self.crop_size]
        # --- END MODIFICATION ---

        return lq_tensor, gt_tensor, lq_name