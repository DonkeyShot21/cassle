import os
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image


class DomainNetDataset(Dataset):
    def __init__(
        self,
        data_root,
        image_list_root,
        domain_names,
        split="train",
        transform=None,
        return_domain=False,
    ):
        self.data_root = data_root
        self.transform = transform
        self.domain_names = domain_names
        self.return_domain = return_domain

        if domain_names is None:
            self.domain_names = [
                "clipart",
                "infograph",
                "painting",
                "quickdraw",
                "real",
                "sketch",
            ]
        if not isinstance(domain_names, list):
            self.domain_name = [domain_names]

        image_list_paths = [
            os.path.join(image_list_root, d + "_" + split + ".txt") for d in self.domain_names
        ]
        self.imgs = self._make_dataset(image_list_paths)

    def _make_dataset(self, image_list_paths):
        images = []
        for image_list_path in image_list_paths:
            image_list = open(image_list_path).readlines()
            images += [(val.split()[0], int(val.split()[1])) for val in image_list]
        return images

    def _rgb_loader(self, path):
        with open(path, "rb") as f:
            with Image.open(f) as img:
                return img.convert("RGB")

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self._rgb_loader(os.path.join(self.data_root, path))

        if self.transform is not None:
            img = self.transform(img)

        domain = None
        if self.return_domain:
            domain = [d for d in self.domain_names if d in path]
            assert len(domain) == 1
            domain = domain[0]

        return domain if self.return_domain else index, img, target

    def __len__(self):
        return len(self.imgs)
