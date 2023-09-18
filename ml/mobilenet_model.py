"""module of mobilenet network."""
import torch.nn as nn
from torchvision.models import mobilenet_v2


class CustomMobileNetv2(nn.Module):
    """class to set up movilenet network."""

    def __init__(
        self, num_class=3, pretrained=True, n_layers=1280, dropout=0.2
    ):
        super().__init__()
        self.mnet = mobilenet_v2(pretrained=pretrained)
        self.freeze()

        self.mnet.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_layers, num_class),
            nn.LogSoftmax(1),
        )

    def forward(self, x):
        return self.mnet(x)

    def freeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.mnet.parameters():
            param.requires_grad = True
