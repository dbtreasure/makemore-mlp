import torch
import numpy as np
import matplotlib.pyplot as plt
import datetime
# summarywriter import
from torch.utils.tensorboard import SummaryWriter

class StepByStep(object):
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device('cpu')
        self.model.to(self.device)
        # The data loader is not a part of the model. It is the input
        # we use to train the model. Since we can specify the model without
        # it, it shouldn't be made an agument of our class
        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.writer = None

        # Variables
        self.losses = []
        self.val_losses = []
        self.test_losses = []
        self.total_epochs = 0

        # Functions
        self.train_step_fn = self._make_train_step_fn()
        self.val_step_fn = self._make_val_step_fn()
        self.test_step_fn = self._make_test_step_fn()

    def to(self, device):
        try:
            self.device = device
            self.model.to(self.device)
        except RuntimeError:
            self.device = ('mps' if getattr(torch,'has_mps',False) \
                else 'gpu' if torch.cuda.is_available() else 'cpu')
            print(f'Could not move to {device}, using {self.device} instead')
            self.model.to(self.device)
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loaders(self, train_loader, val_loader=None, test_loader=None):
        self.train_loader = train_loader
        self.validation_loader = val_loader
        self.test_loader = test_loader
    
    def set_tensorboard(self, name, folder='runs'):
        suffix = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(f'{folder}/{name}-{suffix}')

    # single _ for protected methods, double __ for private methods
    def _make_train_step_fn(self):
        # Builds function that performs a step in the train loop
        
        def perform_train_step_fn(x, y):
            # Set model to training mode
            self.model.train()
            # Forward pass
            yhat = self.model(x)
            # Compute loss
            loss = self.loss_fn(yhat, y)
            self.optimizer.zero_grad()
            # Backward pass
            loss.backward()
            # Update parameters
            self.optimizer.step()
            
            return loss.item()
        
        # Returns the functoin that will be called inside the train loop
        return perform_train_step_fn
    
    def _make_val_step_fn(self):
        # Builds the function that performs a step in the validation loop
        def perform_val_step_fn(x, y):
            # Set model to evaluation mode
            self.model.eval()

            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            # No need to compute Steps 3 and 4, since we are not training
            return loss.item()
        
        return perform_val_step_fn

    def _make_test_step_fn(self):
        # Builds the function that performs a step in the validation loop
        def perform_test_step_fn(x, y):
            # Set model to evaluation mode
            self.model.eval()

            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            # No need to compute Steps 3 and 4, since we are not training
            return loss.item()
        
        return perform_test_step_fn

    def _mini_batch(self, validation=False, test=False):
        # The mini-batch can be used with both loaders
        # The argument validation is used to select the correct loader
        if validation:
            loader = self.validation_loader
            step_fn = self.val_step_fn
        elif test:
            loader = self.test_loader
            step_fn = self.test_step_fn
        else:
            loader = self.train_loader
            step_fn = self.train_step_fn
        
        if loader is None:
            return None
        
        mini_batch_losses = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_loss = step_fn(x_batch, y_batch)
            mini_batch_losses.append(mini_batch_loss)

        loss = np.mean(mini_batch_losses)
        return loss

        
    def set_seed(self, seed=42):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    def train(self, n_epochs, main_tag='training_loss'):
        for epoch in range(n_epochs):
            self.total_epochs += 1
            loss = self._mini_batch(validation=False)
            self.losses.append(loss)
            if self.writer is not None:
                scalars = {f'c_dim:{self.model.C.shape[1]},w1_dim:{self.model.W1.shape[1]},lr:{self.optimizer.state_dict()["param_groups"][0]["lr"]},momentum:{self.optimizer.state_dict()["param_groups"][0]["momentum"]}': loss}

                self.writer.add_scalars(main_tag=main_tag, tag_scalar_dict=scalars, global_step=epoch)
        
        if self.writer:
            self.writer.close()

    def train_validation(self, n_epochs, main_tag='validation_loss'):
        for epoch in range(n_epochs):
            self.total_epochs += 1

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
            
            if self.writer is not None:
                scalars = {f'c_dim:{self.model.C.shape[1]},w1_dim:{self.model.W1.shape[1]},lr:{self.optimizer.state_dict()["param_groups"][0]["lr"]},momentum:{self.optimizer.state_dict()["param_groups"][0]["momentum"]}': val_loss}
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                self.writer.add_scalars(main_tag=main_tag, tag_scalar_dict=scalars, global_step=epoch)
        
        if self.writer:
            self.writer.close()

    def train_test(self, n_epochs, main_tag='test_loss'):
        for epoch in range(n_epochs):
            self.total_epochs += 1

            with torch.no_grad():
                val_loss = self._mini_batch(validation=True)
                self.val_losses.append(val_loss)
            
            if self.writer is not None:
                if val_loss is not None:
                    scalars.update({'validation': val_loss})
                self.writer.add_scalars(main_tag=main_tag, tag_scalar_dict=scalars, global_step=epoch)
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'epoch': self.total_epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.losses,
            'val_loss': self.val_losses,
            'test_loss': self.test_losses
        }
        torch.save(checkpoint, filename)
    
    def load_checkpoint(self, filename):
        checkpoint = torch.load(filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_epochs = checkpoint['epoch']
        self.losses = checkpoint['loss']
        self.val_losses = checkpoint['val_loss']
        self.test_losses = checkpoint['test_loss']
        self.model.train() # always use TRAIN mode after loading a checkpoint

    def plot_losses(self):
        plt.style.use('fivethirtyeight')
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label='Training Loss', c='b')

        if self.validation_loader:
            plt.plot(self.val_losses, label='Validation Loss', c='r')
        
        if self.test_loader:
            plt.plot(self.test_losses, label='Test Loss', c='g')
        
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        return fig

    def add_graph(self):
        if self.train_loader and self.writer:
            x_dummy, y_dummy = next(iter(self.train_loader))
            self.writer.add_graph(self.model, x_dummy.to(self.device))
