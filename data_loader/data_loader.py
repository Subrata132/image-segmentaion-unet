import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class CustomDataLoader(Dataset):
    def __init__(
            self,
            image_path,
            image_transformer=None,
            label_transformer=None
    ):
        self.image_path = image_path
        self.image_names = os.listdir(image_path)[:10]
        self.image_transformer = image_transformer
        self.label_transformer = label_transformer

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(os.path.join(self.image_path, self.image_names[idx])), cv2.COLOR_BGR2RGB)
        image, label = image[:, :256, :], image[:, 256:, :]
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # label = np.transpose(label, (2, 0, 1)).astype(np.float32)
        if self.image_transformer:
            image = self.image_transformer(image)
        if self.label_transformer:
            label = self.label_transformer(label)
        return image, label
