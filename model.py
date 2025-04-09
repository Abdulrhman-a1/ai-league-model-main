import torch.nn as nn
import torchvision.models as models

class CSRNet(nn.Module):
    def __init__(self):
        super(CSRNet, self).__init__()
        from torchvision.models import VGG16_BN_Weights
vgg = models.vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        self.frontend = nn.Sequential(*list(vgg.features.children())[:33])
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        return x
