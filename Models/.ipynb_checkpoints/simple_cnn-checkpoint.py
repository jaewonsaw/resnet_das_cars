import torch
import torch.nn as nn 
import torch.nn.functional as F

class SimpleVehicleNet(nn.Module):
    def __init__(self, num_bins=8, num_count_classes = 7):
        super(SimpleVehicleNet, self).__init__()
        self.num_count_classes = num_count_classes
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
            nn.Linear(32, num_count_classes)
        )

        # Distribution head (predict vehicle count histogram)
        self.dist_head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_bins)  # Output: raw logits → apply log_softmax
        )
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.kl = nn.KLDivLoss(reduction = "batchmean")
        self.lambda_count = 10
        self.lambda_kl = 1
        self.lambda_mse = 1
        self.freq = torch.Tensor([3.8822e-02, 5.4998e-03, 8.6703e-01, 4.8528e-03, 0.0000e+00, 0.0000e+00,
        3.2352e-04, 8.3468e-02])

    def masked_mse(self, y_pred, y_true):
        # Create a mask where target is non-zero
        mask = (y_true != 0).float()
        
        # Compute squared error only where mask == 1
        loss = (mask * (y_pred - y_true) ** 2)
        
        # Avoid division by zero: normalize by number of non-zero elements
        return loss.sum() / (mask.sum() + 1e-8)

    def weighted_mse(self, y_pred, y_true):
        weights = 1.0 / (self.freq + 1e-6)
        weights = weights / weights.sum()
        weights = weights.to(y_pred.device)
        loss = weights * (y_pred - y_true) ** 2
        return loss.mean()
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.flatten(x)
        count = self.count_head(x)                      # shape: [B, 1]
        dist_logits = self.dist_head(x)                # shape: [B, num_bins]
        dist_output = F.softmax(dist_logits, dim=1)
        return count, dist_output

    def loss(self, count_out, hist_out, count_label, hist_label):
        label = count_label.float()
        count_loss = self.cross_entropy(count_out, label)
        row_sums = hist_label.sum(dim=1, keepdim=True)
        normalized = hist_label / row_sums
        hist_loss = self.cross_entropy(hist_out, normalized)
        return self.lambda_count * count_loss + hist_loss
