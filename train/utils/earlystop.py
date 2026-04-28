import os
import torch

class EarlyStopping:
    """
    Early stops the training if validation accuracy doesn't improve after a given patience.
    Monitors a 'max' metric (accuracy); set delta>0 for minimum improvement.
    """
    def __init__(self, patience=5, verbose=False, delta=0.0, path='checkpoint.pt'):
        self.patience = int(patience)
        self.verbose = bool(verbose)
        self.delta = float(delta)
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.is_lr_adjusted = False  # one-shot by default

    def adjust_learning_rate(self, optimizer, min_lr=1e-6, factor=0.1):
        """
        Decrease LR by 'factor' but never below min_lr.
        Returns True if at least one param group LR changed.
        """
        if self.is_lr_adjusted:
            return False

        changed = False
        can_decrease_any = any(pg['lr'] > min_lr for pg in optimizer.param_groups)
        if not can_decrease_any:
            return False

        for pg in optimizer.param_groups:
            old_lr = float(pg['lr'])
            new_lr = max(old_lr * factor, min_lr)
            if new_lr < old_lr - 1e-12:
                pg['lr'] = new_lr
                changed = True
                if self.verbose:
                    print(f'Learning rate decreased: {old_lr:.6f} -> {new_lr:.6f}')

        self.is_lr_adjusted = changed
        return changed

    def __call__(self, acc, model):
        """
        Monitor 'acc' (higher is better). Save best and update early_stop flag.
        'model' should be the UNWRAPPED module in DDP (i.e., model.module).
        """
        score = float(acc)

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Validation accuracy improved ({self.best_score:.6f} -> {score:.6f}). Saving model...')
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        """Save model state_dict safely."""
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        state_dict = getattr(model, "state_dict", None)
        if state_dict is None:
            raise ValueError("model must be a torch.nn.Module (or its .module)")

        torch.save(model.state_dict(), self.path)
        if self.verbose:
            print(f"Saved best model to {self.path}")
