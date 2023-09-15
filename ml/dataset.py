"""class of dataset."""
from PIL import Image
from torch.utils.data import Dataset


class XrayDataset(Dataset):
    """class of dataset to get image and label."""

    def __init__(self, df, transform, img_size, imagenet=True):
        self.imgs_path = df["img_path"]
        self.labels = df["label"]
        self.transform = transform

    def __len__(self):
        """return length of the dataset."""
        return len(self.imgs_path)

    def __get_item__(self, idx):
        """return image, label and path of image corresponding."""
        img_path = self.imgs_path[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = float(self.labels[idx])
        return img, label, img_path
