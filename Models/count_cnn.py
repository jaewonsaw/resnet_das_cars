import torch
import torch.nn as nn
import torch.nn.functional as F


class CountCNN(nn.Module):
    def __init__(self, classes = 7):  # num_bins = number of vehicle count buckets
        super(CountCNN, self).__init__()

        # Input: [B, 1, 585, 130]
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # [B, 16, 585, 130]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 16, 292, 65]

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # [B, 32, 292, 65]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [B, 32, 146, 32]

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # [B, 64, 146, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 64, 1, 1]
        )

        self.flatten = nn.Flatten()  # → [B, 64]

        # Regression head (predict scalar count)
        self.count_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, classes)
        )
        
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.freq = torch.Tensor([3.8822e-02, 5.4998e-03, 8.6703e-01, 4.8528e-03, 0.0000e+00, 0.0000e+00,
        3.2352e-04, 8.3468e-02])
        self.classes = classes
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        count = self.count_head(x)                      # shape: [B, 1]
        return count

    def loss(self, count_out, count_label):
        label = count_label.float()
        count_loss = self.cross_entropy(count_out, count_label)
        return count_loss