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
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os
from sklearn.model_selection import train_test_split
import torchvision

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

class CNN_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding='same')
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(.15)
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding='same')
        self.batch_norm2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding='same')
        self.batch_norm3 = nn.BatchNorm2d(64)
        self.classifier = nn.Sequential(
            nn.Dropout(.5),
            nn.Linear(64 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(.5),
            nn.Linear(64, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.batch_norm1(self.pool(F.relu(self.conv1(x))))
        x = self.dropout(x)
        x = self.batch_norm2(self.pool(F.relu(self.conv2(x))))
        x = self.dropout(x)
        x = self.batch_norm3(self.pool(F.relu(self.conv3(x))))
        x = self.dropout(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = self.classifier(x)
        return x
    
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            m.bias.data.fill_(0.01)
        elif isinstance(m,nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')

class TransformDataset(Dataset):
    def __init__(self, base_dataset, transformations):
        super(TransformDataset, self).__init__()
        self.base = base_dataset
        self.transformations = transformations

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.transformations(x), y
    
# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    return (y_pred.round() == y_true).float().mean()

def prepare_dataset(batch_size=8, data_aug_transform=None, validation_set=True):
    # we use transforms in order to cast from PIL to Tensor datatype and normalize our pixel values in order for them to be in the [-1,1] interval
    transform = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # downloading CIFAR Dataset for training
    train_data = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, 
                                            transform=transform)
    
    # downloading CIFAR Dataset for testing
    test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)

    # splitting into train and validation
    if validation_set:
        train_data, valid_data = train_test_split(train_data, test_size=.2)

    # applying transformations only to the train data using TransformDataset custom class
    if data_aug_transform != None:
        train_data = TransformDataset(train_data, data_aug_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=os.cpu_count())

    val_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,
                                            shuffle=True, num_workers=os.cpu_count())

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=False, num_workers=os.cpu_count())
    if not validation_set:
        return train_loader, test_loader
    
    return train_loader, val_loader, test_loader

def train(dataloader,
          valid_dataloader,
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
    
    history_train_loss = [] 
    history_train_acc = []
    history_val_loss = [] 
    history_val_acc = []
    history = {}

    #####################################################################################
    ################################# TRAIN STEP ########################################
    #####################################################################################

    for epoch in range(n_epochs):
        train_loss = 0.0
        train_acc = 0.0
        for _, (X,y) in enumerate(dataloader, 0):
            
            X,y = X.to(device), y.to(device)
            y_pred = model(X)

            # get metrics for loss
            loss = loss_fn(y_pred, y.long())
            train_loss += loss.item()

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

        train_loss = train_loss/len(dataloader)
        train_acc = train_acc/len(dataloader)

        history_train_loss.append(train_loss)
        history_train_acc.append(train_acc.item())

    #####################################################################################
    ############################### VALIDATION STEP #####################################
    #####################################################################################

        model.eval()

        # Setup test loss and test accuracy values
        val_loss, val_acc = 0, 0

        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for _, (X, y) in enumerate(valid_dataloader):
                # Send data to target device
                X, y = X.to(device), y.to(device)

                # 1. Forward pass
                val_pred = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(val_pred, y)
                val_loss += loss.item()

                # Calculate and accumulate accuracy
                val_pred_labels = val_pred.argmax(dim=1)
                val_acc += accuracy_fn(y,val_pred_labels)
        
        # Adjust metrics to get average loss and accuracy per batch 
        val_loss = val_loss / len(valid_dataloader)
        val_acc = val_acc / len(valid_dataloader)

        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc.item())

        # print metrics each print_interval epochs
        if (epoch + 1) % print_interval == 0:
            print(f"Epoch: {epoch + 1} | \n\t Train Loss: {train_loss:.3} | Train Accuracy: {train_acc:.3}\
                                         \n\t Val Loss: {val_loss:.3} | Val Accuracy: {val_acc:.3}")

    # metrics dumping
    history['train_accuracy'] = history_train_acc
    history['train_loss'] = history_train_loss
    history['val_accuracy'] = history_val_acc
    history['val_loss'] = history_val_loss

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
    plt.plot(history['train_accuracy'], label='train_accuracy')
    plt.plot(history['val_accuracy'], label='val_accuracy')
    plt.title(f'Accuracy for our model. Training for {n_epochs}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10,6))
    plt.plot(history['train_loss'], label='train_loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.title(f'Loss for our model. Training for {n_epochs}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()
