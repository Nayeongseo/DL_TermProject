import torch
import torch.nn as nn
from torchvision.models import alexnet
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        """
        Squeeze-and-Excitation Block.

        Args:
            channels (int): Number of input channels.
            reduction (int): Reduction ratio for bottleneck. Default: 16.
        """
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        # Global Average Pooling
        y = self.global_avg_pool(x).view(batch, channels)
        # Fully Connected Layers
        y = self.fc(y).view(batch, channels, 1, 1)
        # Scale the input
        return x * y

class AlexNet(nn.Module):
    def __init__(self, num_classes=10, reduction=16):
        super(AlexNet, self).__init__()
        
        # Pretrained AlexNet
        pretrained_model = alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")
        
        # Feature extractor with SEBlocks
        self.features = nn.Sequential(
            nn.Sequential(
                pretrained_model.features[0],  # Conv2d(3, 64, kernel_size=11, stride=4, padding=2)
                pretrained_model.features[1],  # ReLU
                pretrained_model.features[2],  # MaxPool2d(kernel_size=3, stride=2)
                SEBlock(channels=64, reduction=reduction)
            ),
            nn.Sequential(
                pretrained_model.features[3],  # Conv2d(64, 192, kernel_size=5, padding=2)
                pretrained_model.features[4],  # ReLU
                pretrained_model.features[5],  # MaxPool2d(kernel_size=3, stride=2)
                SEBlock(channels=192, reduction=reduction)
            ),
            nn.Sequential(
                pretrained_model.features[6],  # Conv2d(192, 384, kernel_size=3, padding=1)
                pretrained_model.features[7],  # ReLU
                SEBlock(channels=384, reduction=reduction)
            ),
            nn.Sequential(
                pretrained_model.features[8],  # Conv2d(384, 256, kernel_size=3, padding=1)
                pretrained_model.features[9],  # ReLU
                SEBlock(channels=256, reduction=reduction)
            ),
            nn.Sequential(
                pretrained_model.features[10],  # Conv2d(256, 256, kernel_size=3, padding=1)
                pretrained_model.features[11],  # ReLU
                pretrained_model.features[12],  # MaxPool2d(kernel_size=3, stride=2)
                SEBlock(channels=256, reduction=reduction)
            )
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        for block in self.features:
            x = block(x)
        x = self.classifier(x)
        return x
