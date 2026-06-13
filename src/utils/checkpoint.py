import torch
import os

def save_checkpoint(model, optimizer, epoch, val_loss, path):
    """Save model state, optimizer state, epoch, loss"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, path)
    print(f"Checkpoint saved: {path} (val_loss: {val_loss:.4f})")

def load_checkpoint(model, optimizer, path):
    """Load checkpoint. Return epoch and val_loss."""
    # weights_only=False locally trust
    checkpoint = torch.load(path, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded: {path}")
    return checkpoint.get('epoch', 0), checkpoint.get('val_loss', float('inf'))
