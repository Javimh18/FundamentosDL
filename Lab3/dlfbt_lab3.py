#===============================================================================
# DLFBT 2023/2024
# Lab assignment 3
# Authors:
#   Manuel Otero NIA1
#   Javier Muñoz Haro NIA2
#===============================================================================

import numpy as np
import tensorflow as tf
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

#===============================================================================
# Exercise 1. Gradient descent to find the minimum of a function
#===============================================================================
def gradient_descent_pytorch(f, x0, learning_rate, niters):
    # Initialize x:
    x_numpy = x0

    # Optimization loop:
    hh = []
    for i in range(niters):
        #-----------------------------------------------------------------------------
        # TO-DO: Define the computational graph using tensors x and y
        #-----------------------------------------------------------------------------
        
        x = torch.tensor(x_numpy, requires_grad=True)
        z = f(x)

        #-----------------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------------
        
        #-----------------------------------------------------------------------------
        # TO-DO: Compute the gradient using tensor dx
        #-----------------------------------------------------------------------------
        
        z.backward()
        dx = x.grad

        #-----------------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------------

        #-----------------------------------------------------------------------------
        # TO-DO: Update x
        #-----------------------------------------------------------------------------
        
        x_numpy = x - learning_rate*dx
        x = x_numpy

        #-----------------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------------

        # Append to history:
        hh.append(x.item())

    return np.array(hh)

#===============================================================================
# Exercise 2. Linear regression using pytorch
#===============================================================================
class LinearRegressionModel_pytorch(object):

    def __init__(self, d=2):
        # Initialize weights and bias:
        self.w = torch.tensor(np.random.normal((d, 1)), requires_grad=True) 
        self.b = torch.tensor(np.random.normal((1, 1)), requires_grad=True) 
        
    def predict(self, x):
        #-----------------------------------------------------------------------
        # TO-DO block: Compute the model output y
        # Note that:
        # - x is a Nxd tensor, with N the number of patterns and d the dimension
        #   (number of features)
        # - y must be a Nx1 tensor
        #-----------------------------------------------------------------------
        
        x = torch.tensor(x, requires_grad=True)
        y = torch.matmul(x, self.w) + self.b

        #-----------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------

        return y


    def compute_gradients(self, x, t):
        #-----------------------------------------------------------------------
        # TO-DO block: Compute the gradients db and dw of the loss function 
        # with respect to b and w
        # Note that:
        # - x is a Nxd tensor, with N the number of patterns and d the dimension
        #   (number of features)
        # - t is a Nx1 tensor
        # - y is a Nx1 tensor
        # - The gradient db (eq. dw) must have the same shape as b (eq. w) 
        #-----------------------------------------------------------------------
        
        loss = self.get_loss(x, t)

        loss.backward()
        db = self.b.grad
        dw = self.w.grad

        #-----------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------
        
        return db, dw
        
    def gradient_step(self, x, t, eta):
        db, dw = self.compute_gradients(x, t)

        #-----------------------------------------------------------------------
        # TO-DO block: Update the model parameters b and w
        #-----------------------------------------------------------------------
        
        self.b = torch.tensor(self.b - eta*db, requires_grad=True)
        self.w = torch.tensor(self.w - eta*dw, requires_grad=True)

        #-----------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------

    def fit(self, x, t, eta, num_iters):
        loss = np.zeros(num_iters)
        for i in range(num_iters):
            self.gradient_step(x, t, eta)
            loss[i] = self.get_loss(x, t).detach().numpy()
        return loss

    def get_loss(self, x, t):
        y = self.predict(x)
        loss = torch.mean(0.5*(y - torch.tensor(t))*(y - torch.tensor(t)))
        return loss
        
        
#===============================================================================
# Exercise 3. Convolutional Neural Network with pytorch
#===============================================================================

#-----------------------------------------------------------------------
# TO-DO: Include here all the code developed for exercise 3, the code
#        should be well documented.
#-----------------------------------------------------------------------

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class CNN_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 4)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 2 * 2, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
         # Using Xavier initilization
        # self.comp_block.apply(init_weights)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x
    
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    return (y_pred.round() == y_true).float().mean()

def train(dataloader,
          model,
          loss_fn, 
          optimizer,
          device, 
          n_epochs = 100,
          print_interval = 2):
    
    """Tests a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    optimizer: function that optimizes the learning of the model
    device: A target device to compute on (e.g. "cuda" or "cpu").
    n_epochs: number of epochs for the model to be trained
    print_inverval: 

    Returns:
    A dictionary with history arrays for training loss and training accuracy metrics.
    """
    
    history_loss = [] 
    history_acc = []
    history = {}

    for epoch in range(n_epochs):
        train_loss = 0.0
        train_acc = 0.0
        for _, (X,y) in enumerate(dataloader, 0):
            
            X,y = X.to(device), y.to(device)
            y_pred = model(X)

            # get metrics for loss
            loss = loss_fn(y_pred, y.long())
            train_loss += loss

            # get metrics for accuracy
            y_pred_class = torch.argmax(y_pred, dim=1)
            acc = accuracy_fn(y_true=y, y_pred=y_pred_class)
            train_acc += acc

            # reseting the optimizer
            optimizer.zero_grad()
            # backward pass (backpropagation)
            loss.backward()
            # adjusting the necessary weigths of our models
            optimizer.step()

        epoch_loss = train_loss/len(dataloader)
        epoch_acc = train_acc/len(dataloader)

        if (epoch + 1) % print_interval == 0:
            print(f"Epoch: {epoch + 1} | Loss: {epoch_loss} | Accuracy: {epoch_acc}")

        history_acc.append(epoch_acc.item())
        history_loss.append(epoch_loss.item())

        history['accuracy'] = history_acc
        history['loss'] = history_loss

    return history

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:

    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval() 

  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0

  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)

          # 1. Forward pass
          test_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()

          # Calculate and accumulate accuracy
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def plot_metrics(history:dict, n_epochs:int):
    # plotting training statistics

    plt.figure(figsize=(10,6))
    plt.plot(history['accuracy'])
    plt.title(f'Accuracy for our basic model. Training for {n_epochs}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(history['loss'])
    plt.title(f'Loss for our basic model. Training for {n_epochs}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()
