import os
import torch
import tifffile as tiff
import contextlib
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


class MultiSpectralDataset(Dataset):
    def __init__(self, root_dir, img_size):
        self.samples = []
        self.classes = sorted(os.listdir(root_dir))
        self.img_size = img_size

        for label, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            for file in os.listdir(cls_path):
                self.samples.append((os.path.join(cls_path, file), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stderr(fnull):
                img = tiff.imread(path)

        # Convert to float tensor
        img = torch.tensor(img.astype("float32")).permute(2, 0, 1)

        # Normalize
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

        # Resize
        img = F.interpolate(
            img.unsqueeze(0),
            size=(self.img_size, self.img_size),
            mode='bilinear'
        ).squeeze(0)

        return img, label


def get_loaders(path, batch_size, img_size):

    train = MultiSpectralDataset(os.path.join(path, "train"), img_size)
    val   = MultiSpectralDataset(os.path.join(path, "val"), img_size)
    test  = MultiSpectralDataset(os.path.join(path, "test"), img_size)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val, batch_size=batch_size)
    test_loader  = DataLoader(test, batch_size=batch_size)

    return train_loader, val_loader, test_loader, train.classes