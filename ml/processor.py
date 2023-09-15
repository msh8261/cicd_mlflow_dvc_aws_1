"""transform images in dataset."""
from torchvision import transforms


def get_preprocessor(size=(160, 160), imagenet=True):
    """return transformed image."""
    if imagenet:
        train_transform = transforms.Compose(
            [
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize(230),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0, 5], std=[0.5, 0.5, 0, 5]
                ),
            ]
        )
        val_transform = transforms.Compose(
            [
                transforms.Resize(size=size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0, 5], std=[0.5, 0.5, 0, 5]
                ),
            ]
        )

    return train_transform, val_transform
