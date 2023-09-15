"""class of classifier."""
import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from ml.mobilenet_model import CustomMobileNetv2


def get_accuracy(y_pred: torch.Tensor, y_test: torch.Tensor):
    """calculate the accuracy of the model."""
    return ((y_pred > 0.0) == y_test).float().mean()


class XrayClassifier(pl.LightningModule):
    """class to set up automatically to train and test."""

    def __init__(self, imagenet_weights=True, dropout=0.0, lr=0.001):
        super().__init__()
        self.classifier = CustomMobileNetv2(
            num_class=3, pretrained=imagenet_weights, dropout=dropout
        )
        self.lr = lr

    def forward(self, x):
        """apply classifier."""
        return self.classifier(x)

    def configure_optimizers(self):
        """configure the optimizer."""
        return torch.optim.Adam(self.parameters(), self.lr)

    def __compute_batch_loss(self, batch):
        """compute the loss and accuracy and return it."""
        x, y, _ = batch
        y = y.unsqueeze(axis=1)
        x = self.classifier(x)
        weights = torch.Tensor([1.0, 1.0, 1.0])
        loss = F.cross_entropy(x, y, weights)
        batch_size = len(x)
        acc = get_accuracy(x, y)
        return loss, acc, batch_size

    def training_step(self, train_batch, batch_idx):
        """set train step to log information."""
        loss, accuracy, batch_size = self.__compute_batch_loss(train_batch)
        self.log("train_loss", loss, batch_size=batch_size)
        self.log("train_accuracy", accuracy, batch_size=batch_size)
        return loss

    def validation_step(self, val_batch, batch_idx):
        """set val step to log information."""
        loss, accuracy, batch_size = self.__compute_batch_loss(val_batch)
        self.log("val_loss", loss, batch_size=batch_size)
        self.log("val_accuracy", accuracy, batch_size=batch_size)

    def test_step(self, test_batch, batch_idx):
        """set test step to log information."""
        loss, accuracy, batch_size = self.__compute_batch_loss(test_batch)
        self.log("test_loss", loss, batch_size=batch_size)
        self.log("test_accuracy", accuracy, batch_size=batch_size)
