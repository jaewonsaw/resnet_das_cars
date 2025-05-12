import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import Logger

class Trainer:
    def __init__(self, model, opt):
        self.model = model
        self.opt = opt
    
    def train(self, epochs, log_dir, train_loader, val_loader):
        logger = Logger(log_dir)
        best_val_loss = float('inf')
        logger.save_checkpoint("best_model.pth", self.model, optimizer, 0, best_val_loss)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for epoch in tqdm(range(num_epochs)):
            self.model.train()
            train_loss = 0.0
            train_count_correct = 0
            train_label_correct = 0
            train_total = 0
            
            for images, labels, counts in train_loader:
                images = images.to(device).float()
                labels = labels.to(device).float()
                counts = counts.to(device).long()
                self.opt.zero_grad()
                count_output, label_output = self.model(images)
                loss = self.model.loss(count_output, label_output, counts, labels)
                loss.backward()
                self.opt.step()

                train_loss += loss.item()
                preds = torch.argmax(count_output, dim = -1)
                label_preds = preds.unsqueeze(-1) * label_output
                train_count_correct += (preds == counts).sum().item()
                train_label_correct += (torch.round(label_preds).flatten() == labels.flatten()).sum().item()
                train_total += labels.size(0)

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            train_acc_count = train_count_correct / train_total
            train_acc_label = train_label_correct / (8*train_total)

            # --------- Validate ---------
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_label_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels, counts in val_loader:
                    images = images.to(device).float()
                    labels = labels.to(device).float()
                    counts = counts.to(device).long()#.unsqueeze(1)
                    count_output, label_output = self.model(images)
                    loss = self.model.loss(count_output, label_output, counts, labels)
                    val_loss += loss.item()
                    preds = torch.argmax(count_output, dim = -1)
                    label_preds = preds.unsqueeze(-1) * label_output
                    val_correct += (preds == counts.flatten()).sum().item()
                    val_label_correct += (torch.round(label_output).flatten() == labels.flatten()).sum().item()
                    val_total += labels.size(0)


            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_acc = val_correct / val_total
            val_acc_label = val_label_correct / (8*val_total)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                logger.save_checkpoint("best_model.pth", self.model, optimizer, epoch, best_val_loss)

            # --------- Print results ---------

            logger.log(f"Epoch {epoch+1}/{num_epochs} "
                  f"Train Loss: {avg_train_loss:.2f} | Count Acc: {train_acc_count:.2f} | Label Acc: {train_acc_label:.2f}"
                  f"|| Val Loss: {avg_val_loss:.2f} | Count Acc: {val_acc:.2f} | Label Acc: {val_acc_label:.2f}")
        return train_losses, val_losses
