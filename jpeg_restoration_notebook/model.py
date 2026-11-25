"""Small fully convolutional residual restoration CNN."""
import torch
import torch.nn as nn

class ResidualRestorationCNN(nn.Module):
    """
    Small residual CNN for JPEG artifact removal.
    Predicts a correction residual and adds it to the compressed image.
    """
    def __init__(self, channels=32, depth=5):
        super().__init__()
        if channels < 1 or depth < 2:
            raise ValueError('channels >= 1 and depth >= 2 are required')
        layers = [nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(channels, 3, 3, padding=1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        residual = self.net(x)
        return torch.clamp(x + residual, 0.0, 1.0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
