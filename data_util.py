import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os


class VesselDataset(Dataset):
    def __init__(self, root_path, label_mode='1', augmentation=False, blue=False):
        files = os.listdir(root_path)
        files.sort()
        i = 0
        self.data = []
        while i < len(files):
            raw_img = Image.open(os.path.join(root_path, files[i]))
            ho1 = Image.open(os.path.join(root_path, files[i + 1])).convert('1')
            ho2 = Image.open(os.path.join(root_path, files[i + 2])).convert('1')
            transform = transforms.Compose([
                transforms.Resize((480, 480)),
                transforms.ToTensor(),
            ])
            if augmentation:

                aug = transforms.Compose([
                    transforms.ColorJitter()
                ])
                raw_img = aug(raw_img)
            raw_img = transform(raw_img)

            if blue:
                b = raw_img[2, :, :]
                min_gl, max_gl = b.min(), b.max()
                b[:] = (b - min_gl) / (max_gl - min_gl)

            ho1 = transform(ho1)
            ho2 = transform(ho2)
            i += 3
            if label_mode == '1':
                self.data.append((raw_img, ho1))
            elif label_mode == '2':
                self.data.append((raw_img, ho2))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


if __name__ == '__main__':
    ds = VesselDataset('data/CHASEDB1')
    print(ds[0])
    dl = DataLoader(ds)
    for raw, label in dl:
        print(raw.shape, label.shape)
        break
