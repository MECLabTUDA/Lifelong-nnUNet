import torch
import numpy as np
class EarlyStop:
    """Used to early stop the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0, 
                 save_name="checkpoint.pt", save_callback=None,
                 trainer=None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            save_name (string): The filename with which the model and the optimizer is saved when improved.
                            Default: "checkpoint.pt"
        """
        self.patience = patience
        self.verbose = verbose
        self.save_name = save_name
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_callback = save_callback
        self.trainer = trainer
        assert(self.delta == 0)

    def __call__(self, val_loss):
        print(self.best_loss)
        print(val_loss)
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss)
        elif val_loss > self.best_loss:
            self.trainer.print_to_log_file(f"EarlyStopping: {val_loss} > {self.best_loss}")
            self.counter += 1
            self.trainer.print_to_log_file(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.trainer.print_to_log_file(f"EarlyStopping: {val_loss} <= {self.best_loss}")
            self.best_loss = val_loss
            self.save_checkpoint(val_loss)
            self.counter = 0
            
        return self.early_stop

    def save_checkpoint(self, val_loss):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trainer.print_to_log_file(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.val_loss_min = val_loss
        if self.save_callback is not None:
            self.save_callback()