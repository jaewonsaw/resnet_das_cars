from resnet import Resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F

class VehicleCounterNet(nn.Module):
    def __init__(self, num_classes=5, count_classes = 7, p = 0.3):
        super(VehicleCounterNet, self).__init__()

        # Load pretrained ResNet18 and modify
        self.backbone = Resnet18()
        num_features = 512
        # Output head 1: Regress or classify number of vehicles
        self.vehicle_count = nn.Sequential(
                                           nn.Linear(num_features, num_features//2),
                                           nn.BatchNorm1d([num_features//2]),
                                           nn.Dropout(p),
                                           nn.ReLU(),
                                           nn.Linear(num_features//2, count_classes))

        # Output head 2: Predict histogram (e.g., soft count distribution across possible values)
        self.histogram_head = nn.Sequential(nn.Linear(num_features, num_features//2),
                                           nn.BatchNorm1d([num_features//2]),
                                           nn.ReLU(),
                                           nn.Linear(num_features//2, num_classes))
        self.cross_entropy = nn.CrossEntropyLoss()
        self.freq = torch.Tensor([3.8822e-02, 5.4998e-03, 8.6703e-01, 4.8528e-03, 8.3468e-02])
        weights = 1e-12 + self.freq # [1, num_classes]
        weights = 1/weights
        self.weights = torch.clip(weights, 1, 2)
        #lambda P, Q: -torch.sum(self.freq*P*torch.log(Q + 1e-9))
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction = "batchmean")
        self.lambda_count = 1
        self.lambda_kl = 10
        self.lambda_mse = 1
        self.count_classes = count_classes

    def forward(self, x):
        features = self.backbone(x)
        count_output = self.vehicle_count(features)
        histogram_output = F.softmax(self.histogram_head(features), dim = 1)
        return count_output, histogram_output

    def masked_mse(self, y_pred, y_true):
        # Create a mask where target is non-zero
        mask = (y_true != 0).float()
        
        # Compute squared error only where mask == 1
        loss = (mask * (y_pred - y_true) ** 2)
        
        # Avoid division by zero: normalize by number of non-zero elements
        return loss.sum() / (mask.sum() + 1e-8)
    
    def weighted_soft_nll_loss(self, log_probs, soft_targets):
        """
        log_probs: Tensor of shape [batch_size, num_classes], output of log_softmax
        soft_targets: Tensor of shape [batch_size, num_classes], target distributions
        class_weights: Tensor of shape [num_classes], weight per class
        """
        weights = self.weights.unsqueeze(0).to(log_probs.device)  # [1, num_classes]
        
        # Apply weights to soft_targets
        weighted_targets = soft_targets * weights  # [batch_size, num_classes]
        
        # Compute element-wise product: -weighted_targets * log_probs
        loss = -torch.sum(weighted_targets * log_probs, dim=1)  # [batch_size]
        
        return loss.mean()  # or use .sum() if preferred

    def loss(self, count_out, hist_out, count_label, hist_label):
        count_label = F.one_hot(count_label, self.count_classes).float()
        count_loss = self.cross_entropy(count_out, count_label)
        target = hist_label + 1e-8
        target_probs = (target) / target.sum(dim=1, keepdim=True)
        expected_hist_loss = self.weighted_soft_nll_loss(torch.log(hist_out), target_probs)
        #mse_loss = self.mse(count_label * hist_label, torch.argmax(count_out, dim = -1).unsqueeze(-1) * hist_out)
        total_loss = (
            self.lambda_count * count_loss +
            self.lambda_kl * expected_hist_loss
         #   self.lambda_mse * mse_loss
        )
        
        return total_loss