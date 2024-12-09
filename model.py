import torch
import torch.nn as nn
import torchvision.models as models

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Global Average Pooling
        batch_size, channels, _, _ = x.size()
        se_weight = self.global_avg_pool(x).view(batch_size, channels)
        # Fully Connected layers
        se_weight = self.fc(se_weight).view(batch_size, channels, 1, 1)
        # Scale input
        return x * se_weight


class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        # Load ResNet50 as backbone
        self.resnet = models.resnet50(pretrained=True)
        
        # Remove the original fully connected layer
        self.features = nn.Sequential(*list(self.resnet.children())[:-2])  # Up to the last Conv layer
        
        # SE Attention Block
        self.attention = SEBlock(in_channels=2048, reduction=16)
        
        # Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Feature Extraction
        x = self.features(x)
        
        # Attention Mechanism
        x = self.attention(x)
        
        # Pooling and Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class VGGNet(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGNet, self).__init__()
        # Load VGG16 as backbone
        self.vgg = models.vgg16(pretrained=True)
        
        # Remove the classifier layers
        self.features = self.vgg.features  # Only convolutional layers
        
        # SE Attention Block
        self.attention = SEBlock(in_channels=512, reduction=16)  # VGG16 final conv output channels are 512
        
        # Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    

class GoogleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet, self).__init__()
        # Load GoogleNet as backbone
        self.googlenet = models.googlenet(pretrained=True)
        
        # Remove the fully connected layer
        self.features = nn.Sequential(*list(self.googlenet.children())[:-2])  # Up to the last Conv layer
        
        # SE Attention Block
        self.attention = SEBlock(in_channels=1024, reduction=16)  # GoogleNet final conv output channels are 1024
        
        # Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x