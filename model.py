import torch
import torch.nn as nn
from torchvision import models

class MultiTaskFaceNet(nn.Module):
    def __init__(self, arch='efficientnet_b0'):
        super(MultiTaskFaceNet, self).__init__()
        # Backbone - EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights='DEFAULT')
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()

        # Жас тармағы (Қате бойынша: Linear -> BatchNorm -> Dropout -> Linear -> Sigmoid)
        self.age_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        # Жыныс тармағы (Қате бойынша: Linear -> BatchNorm -> Dropout -> Linear)
        self.gender_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        features = self.backbone(x)
        age = self.age_head(features)
        gender = self.gender_head(features)
        return age, gender