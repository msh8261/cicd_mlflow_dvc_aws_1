"""class of classifier."""
import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import functional as F

from ml.mobilenet_model import CustomMobileNetv2


class XrayClassifier(pl.LightningModule):
    """class to set up automatically to train and test."""

    def __init__(self, imagenet_weights=True, dropout=0.0, lr=0.001):
        super().__init__()
        self.classifier = CustomMobileNetv2(
            num_class=3, pretrained=imagenet_weights, dropout=dropout
        )
        self.lr = lr
        self.critrion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        """apply classifier."""
        return self.classifier(x)

    def configure_optimizers(self):
        """configure the optimizer."""
        return torch.optim.Adam(
            self.parameters(), self.lr, weight_decay=0.00001
        )

    def __compute_batch_loss(self, batch):
        """compute the loss and accuracy and return it."""
        x, y, _ = batch
        y = y.unsqueeze(axis=1)
        # y = y.long()
        cls = self(x)
        cls = F.softmax(cls.float(), dim=1)
        y_pred = cls.data.max(dim=1)[1].unsqueeze(axis=1)
        loss = self.critrion(y_pred.float(), y.float())
        loss.requires_grad = True
        loss.backward()
        batch_size = len(y_pred)
        acc = torchmetrics.functional.accuracy(
            y_pred, y, task="multiclass", num_classes=3
        )
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
