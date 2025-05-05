import os
from datetime import datetime
import torch

class Logger:
    def __init__(self, log_dir, filename='train_log.txt'):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.log_dir = log_dir
        self.log_path = os.path.join(log_dir, f'{timestamp}_{filename}')

        with open(self.log_path, 'w') as f:
            f.write(f"Logging started: {timestamp}\n\n")

    def log(self, message):
        timestamp = datetime.now().strftime('%H:%M:%S')
        full_message = f"[{timestamp}] {message}"
        print(full_message)  # also print to stdout
        with open(self.log_path, 'a') as f:
            f.write(full_message + '\n')

    def log_metrics(self, epoch, train_loss=None, val_loss=None, **kwargs):
        msg = f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        for k, v in kwargs.items():
            msg += f" | {k}: {v:.4f}" if isinstance(v, float) else f" | {k}: {v}"
        self.log(msg)

    def save_checkpoint(self, path, model, optimizer, epoch, best_val_loss):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint, os.path.join(self.log_dir, path))
