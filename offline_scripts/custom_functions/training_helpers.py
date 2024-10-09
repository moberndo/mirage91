import numpy as np


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience  # Number of epochs to wait before stopping if no improvement
        self.min_delta = min_delta  # Minimum change to count as improvement
        self.counter = 0  # Count epochs with no improvement
        self.best_loss = None  # Best validation loss observed
        self.acc = None  # Accuracy corresponding to best loss
        self.early_stop = False  # Flag to indicate stopping

    def __call__(self, val_loss, val_acc):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.acc = val_acc
        elif val_loss > self.best_loss + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.acc = val_acc
            self.counter = 0

def count_parameters(model):
    return sum(np.prod(p.size()) for p in model.parameters())