import torch
import torch.nn as nn
from torchvision.models import alexnet

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        # Pretrained AlexNet
        pretrained_model = alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")
        
        # Extract the feature extractor
        self.features = pretrained_model.features
        
        # Replace the classifier with a custom one
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.BatchNorm1d(4096),  # Add Batch Normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(4096, 4096),
            nn.BatchNorm1d(4096),  # Add Batch Normalization
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(4096, num_classes)  # Change output to match `num_classes`
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
