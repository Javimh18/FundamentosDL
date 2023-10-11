#===============================================================================
# DLFBT 2023/2024
# Lab assignment 3
# Authors:
#   Name1 NIA1
#   Name2 NIA2
#===============================================================================

import numpy as np
import tensorflow as tf
import torch

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


