""" VIIRS Vessel Dataset
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_LABELS = {"incorrect": 0, "correct": 1}


class VIIRSVesselDataset(Dataset):
    """Dataset for VIIRS Vessel Detection"""

    def __init__(self, root_dir: str, transform: transforms = None):
        """Dataset for VIIRS Vessel Detection

        Parameters
        ----------
        root_dir : str
            _description_
        transform : transforms, optional
            _description_, by default None
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = list(Path(root_dir).rglob("*.npy"))
        self.class_map = CLASS_LABELS
        self.targets = [self.class_map[img_name.parts[-2]] for img_name in self.images]

    def __len__(self) -> int:
        """returns length of dataset

        Returns
        -------
        int

        """
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, torch.tensor]:
        """gets item from dataset

        Parameters
        ----------
        idx : Union[slice, int]

        Returns
        -------
        Tuple[np.ndarray, torch.tensor]
        """
        img_name = self.images[idx]
        self.image = np.load(img_name.resolve()).astype(np.float32)
        label = img_name.parts[-2]
        self.class_id = torch.tensor(self.class_map[label])
        if self.transform:
            self.transform(self.image)
        return self.image, self.class_id
