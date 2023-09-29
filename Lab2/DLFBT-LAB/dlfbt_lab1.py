#===============================================================================
# DLFBT 2023/2024
# Lab assignment 1
# Authors:
#   Name1 NIA1
#   Name2 NIA2
#===============================================================================

import numpy as np
import tensorflow as tf

#===============================================================================
# Exercise 1. Linear regression
#===============================================================================
class LinearRegressionModel(object):

    def __init__(self, d=2):
        # Initialize weights and bias:
        self.w = np.random.randn(d, 1)
        self.b = np.random.randn(1, 1)
        
    def predict(self, x):
        #-----------------------------------------------------------------------
        # TO-DO block: Compute the model output y
        # Note that:
        # - x is a Nxd array, with N the number of patterns and d the dimension
        #   (number of features)
        # - y must be a Nx1 array
        #-----------------------------------------------------------------------
        pass
        #-----------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------

        return y

    def compute_gradients(self, x, t):
        y = self.predict(x)

        #-----------------------------------------------------------------------
        # TO-DO block: Compute the gradients db and dw of the loss function 
        # with respect to b and w
        # Note that:
        # - x is a Nxd array, with N the number of patterns and d the dimension
        #   (number of features)
        # - t is a Nx1 array
        # - y is a Nx1 array
        # - The gradient db (eq. dw) must have the same shape as b (eq. w) 
        #-----------------------------------------------------------------------
        pass
        #-----------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------
        
        return db, dw
        
    def gradient_step(self, x, t, eta):
        db, dw = self.compute_gradients(x, t)
        
        #-----------------------------------------------------------------------
        # TO-DO block: Update the model parameters b and w
        #-----------------------------------------------------------------------
        pass
        #-----------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------

    def fit(self, x, t, eta, num_iters):
        loss = np.zeros(num_iters)
        for i in range(num_iters):
            self.gradient_step(x, t, eta)
            loss[i] = self.get_loss(x, t)
        return loss

    def get_loss(self, x, t):
        y = self.predict(x)
        loss = 0.5*np.mean((y - t)**2)
        return loss


#===============================================================================
# Exercise 2. Logistic regression
#===============================================================================
class LogisticRegressionModel(LinearRegressionModel):
    def __init__(self, d=2):
        LinearRegressionModel.__init__(self, d)

    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    #---------------------------------------------------------------------------
    # TO-DO block: Overwrite the methods of the LinearRegressionModel class
    #---------------------------------------------------------------------------
    pass
    #---------------------------------------------------------------------------
    # End of TO-DO block 
    #---------------------------------------------------------------------------
    
    def get_loss(self, x, t):
        y = self.predict(x)
        loss = -np.mean(t*np.log(y) + (1.-t)*np.log(1.-y))
        return loss


#===============================================================================
# Exercise 3. Use TensorFlow to evaluate the derivative of a function
#===============================================================================
def derivative(f, x):
  x = tf.Variable(x)

  #-----------------------------------------------------------------------------
  # TO-DO block: Define the computational graph within a gradient tape and
  # compute the gradient
  #-----------------------------------------------------------------------------
  pass
  #-----------------------------------------------------------------------------
  # End of TO-DO block
  #-----------------------------------------------------------------------------

  return dy_dx


#===============================================================================
# Exercise 4. Gradient descent to find the minimum of a function
#===============================================================================
def gradient_descent(f, x0, learning_rate, niters):
    # Initilize x:
    x = tf.Variable(x0)

    # List to store the x values:
    x_history = []

    # Optimization loop:
    for i in range(niters):
        #-----------------------------------------------------------------------------
        # TO-DO block: Define the computational graph within a gradient tape and 
        # compute the gradient
        #-----------------------------------------------------------------------------
        pass
        #-----------------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------------

        #-----------------------------------------------------------------------------
        # TO-DO block: Update the value of x using the tf.Variable assign method
        #-----------------------------------------------------------------------------
        pass
        #-----------------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------------

        # Append x to history:
        x_history.append(x.numpy())

    return np.array(x_history)


#===============================================================================
# Exercise 5. Linear regression using tensorflow
#===============================================================================
class LinearRegressionModel_TF(object):

    def __init__(self, d=2):
        # Initialize weights and bias:
        self.w = tf.Variable(tf.random.normal(shape=[d, 1], dtype=tf.dtypes.float64))  
        self.b = tf.Variable(tf.random.normal(shape=[1, 1], dtype=tf.dtypes.float64)) 
        
    def predict(self, x):
        #-----------------------------------------------------------------------
        # TO-DO block: Compute the model output y
        # Note that:
        # - x is a Nxd tensor, with N the number of patterns and d the dimension
        #   (number of features)
        # - y must be a Nx1 tensor
        #-----------------------------------------------------------------------
        pass
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
        pass
        #-----------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------
        
        return db, dw
        
    def gradient_step(self, x, t, eta):
        db, dw = self.compute_gradients(x, t)
        
        #-----------------------------------------------------------------------
        # TO-DO block: Update the model parameters b and w
        #-----------------------------------------------------------------------
        pass
        #-----------------------------------------------------------------------
        # End of TO-DO block 
        #-----------------------------------------------------------------------

    def fit(self, x, t, eta, num_iters):
        loss = np.zeros(num_iters)
        for i in range(num_iters):
            self.gradient_step(x, t, eta)
            loss[i] = self.get_loss(x, t).numpy()
        return loss

    def get_loss(self, x, t):
        y = self.predict(x)
        loss = tf.reduce_mean(0.5*(y - t)*(y - t))
        return loss


#===============================================================================
# Exercise 6. Implementation of a neural network with Numpy
#===============================================================================
class NeuralNetwork(object):

    #---------------------------------------------------------------------------
    # Sigmoid function:
    #---------------------------------------------------------------------------
    def sigmoid(z):
        return 1.0/(1.0+np.exp(-z))

    #---------------------------------------------------------------------------
    # Derivative of the sigmoid function:
    #---------------------------------------------------------------------------
    def dsigmoid(z):
        return NeuralNetwork.sigmoid(z)*(1.0 - NeuralNetwork.sigmoid(z))

    #---------------------------------------------------------------------------
    # Identity function:
    #---------------------------------------------------------------------------
    def identity(z):
        return z

    #---------------------------------------------------------------------------
    # Derivative of the identity function:
    #---------------------------------------------------------------------------
    def didentity(z):
        return np.ones_like(z)

    #---------------------------------------------------------------------------
    # MSE loss:
    #---------------------------------------------------------------------------
    def mse_loss(y, t):
        loss = 0.5*np.mean((y - t)**2)
        return loss

    #---------------------------------------------------------------------------
    # Cross-entropy loss:
    #---------------------------------------------------------------------------
    def cross_entropy_loss(y, t):
        loss = -np.mean(t*np.log(y) + (1.-t)*np.log(1.-y))
        return loss

    #---------------------------------------------------------------------------
    # The constructor receives a list of tuples of the form (n, a), one tuple
    # for each layer, where n is the number of units in the layer and a is the
    # type of layer. The following types are allowed:
    #
    # - 'sigmoid': a layer of sigmoid units
    # - 'linear': a layer of linear units
    # - 'input': a special layer for the input
    #
    # The minimum is two layers (input and output).
    # The first layer must be always of type 'input'.
    # Only one 'input' layer must be present, and it must be the first layer.
    # The default value is layers=[(2, 'input'), (1, 'sigmoid')].
    # The output layer must have one single neuron.
    #---------------------------------------------------------------------------
    def __init__(self, layers=[(2, 'input'), (1, 'sigmoid')]):
        # Set the number of layers (without the input layer):
        self.nlayers = len(layers) - 1

        # The network weights and biases are stored in lists self.W and self.b
        # of length self.nlayers. For each layer, the activation function and
        # its derivative are stored in lists self.a and self.da:
        self.W = []
        self.b = []
        self.a = []
        self.da = []

        # This loop sets the weights, biases and activation functions for each
        # layer:
        for l0, l1 in zip(layers[:-1], layers[1:]):
            self.W.append(np.random.randn(l1[0], l0[0]))
            self.b.append(np.random.randn(l1[0], 1))
            if l1[1] == 'sigmoid':
                self.a.append(NeuralNetwork.sigmoid)
                self.da.append(NeuralNetwork.dsigmoid)
            elif l1[1] == 'linear':
                self.a.append(NeuralNetwork.identity)
                self.da.append(NeuralNetwork.didentity)

    #---------------------------------------------------------------------------
    # Implementation of the forward pass, compute the network output y given the
    # input x.
    #
    # Input: x is a numpy array of dimension dxN, with N the number of patterns
    #        and d the dimension (number of features)
    #
    # Return: z is a list with the pre-activations of each layer: z[i] must be
    #         an array of shape nixN with ni the number of neurons in layer i
    #         y is a list with the activations of each layer: y[i] must be
    #         an array of shape nixN with ni the number of neurons in layer i
    #---------------------------------------------------------------------------
    def predict(self, x):
        # Create empty lists for z and y:
        z = []
        y = []
        #-----------------------------------------------------------------------
        # TO-DO block: loop in the network layers computing both the pre-
        # activation and the activation and appending them to lists z and y.
        #-----------------------------------------------------------------------
        pass
        #-----------------------------------------------------------------------
        # End of TO-DO block
        #-----------------------------------------------------------------------

        return z, y

    #---------------------------------------------------------------------------
    # Implementation of the backward pass, compute the gradients of the loss
    # function with respect to all the weights and biases.
    #
    # Input: x is a numpy array of dimension dxN, with N the number of patterns
    #        and d the dimension (number of features).
    #        t is a numpy array of dimension 1xN, with N the number of patterns.
    #
    # Return: dW is a list with the gradients of the loss function with respect
    #         to the weights of each layer
    #         db is a list with the gradients of the loss function with respect
    #         to the biases of each layer
    #---------------------------------------------------------------------------
    def compute_gradients(self, x, t):
        # Get the number of patterns:
        n = x.shape[1]

        # Create empty lists for dW and db:
        dW = []
        db = []

        # Call the predict method (forward pass). The preactivations z and the
        # activations y in each layer are needed for the backward pass:
        z, y = self.predict(x)

        # Derivative of the loss with respect to z in the last layer. This is
        # valid both for a regression problem with linear output and MSE loss,
        # and for a classification problem with sigmoid output and cross-entropy
        # loss:
        dy = (y[-1] - t) / n

        #-----------------------------------------------------------------------
        # TO-DO block: loop in the network layers computing the gradients with
        # respect to W and b.
        #
        # Note that the gradients must be computed starting by the last layer,
        # it may be useful to traverse the lists backwards.
        #-----------------------------------------------------------------------
        pass
        #-----------------------------------------------------------------------
        # End of TO-DO block
        #-----------------------------------------------------------------------

        return dW, db

    #---------------------------------------------------------------------------
    # Gradient step:
    #---------------------------------------------------------------------------
    def gradient_step(self, x, t, eta):
        dW, db = self.compute_gradients(x, t)

        #-----------------------------------------------------------------------
        # TO-DO block: Loop in layers updating the model parameters b and w
        #-----------------------------------------------------------------------
        pass
        #-----------------------------------------------------------------------
        # End of TO-DO block
        #-----------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # Fit the model parameters to the training data x, t.
    # Return the loss at each training epoch.
    #---------------------------------------------------------------------------
    def fit(self, x, t, eta, num_epochs, batch_size, loss_function):
        (dim, n) = x.shape
        num_batches = (n // batch_size) + ((n % batch_size) != 0)

        loss = np.zeros(num_epochs)
        for i in range(num_epochs):
            # Shuffle data and generate batches:
            ix = np.random.permutation(n)
            for j in range(num_batches):
                imin = j*batch_size
                imax = np.minimum((j+1)*batch_size, n)

                ibatch = ix[imin:imax]
                batch_x = x[:, ibatch]
                batch_t = t[:, ibatch]
                self.gradient_step(batch_x, batch_t, eta)

            # At the end of each epoch, compute the model loss on all data:
            loss[i] = self.get_loss(x, t, loss_function)
        return loss

    #---------------------------------------------------------------------------
    # Compute loss:
    #---------------------------------------------------------------------------
    def get_loss(self, x, t, loss_function):
        _, y = self.predict(x)
        return loss_function(y[-1], t)


#===============================================================================
# Exercise 7. Implementation of a neural network with TensorFlow
#===============================================================================
class NeuralNetwork_TF(object):
    #---------------------------------------------------------------------------
    # MSE loss:
    #---------------------------------------------------------------------------
    def mse_loss(y, t):
        loss = 0.5*tf.reduce_mean(tf.pow(y-t, 2.0))
        return loss
    #---------------------------------------------------------------------------
    # Cross-entropy loss:
    #---------------------------------------------------------------------------
    def cross_entropy_loss(y, t):
        loss = -tf.reduce_mean(t*tf.math.log(y) + (1.-t)*tf.math.log(1.-y))
        return loss

    #---------------------------------------------------------------------------
    # The constructor receives a list of tuples of the form (n, a), one tuple
    # for each layer, where n is the number of units in the layer and a is the
    # activation of the layer. You may use any TensorFlow activation. In
    # particular:
    #
    # - tf.sigmoid: sigmoid activation function
    # - tf.identity: identity activation function (to implement a linear layer)
    #
    # The activation function is ignored for the input layer.
    #
    # The minimum is two layers (input and output).
    # The default value is layers=[(2, None), (1, tf.sigmoid)].
    # The output layer must have one single neuron.
    #---------------------------------------------------------------------------
    def __init__(self, layers=[(2, None), (1, tf.sigmoid)]):
        # Network weights and activations:
        self.nlayers = len(layers) - 1
        self.W = []
        self.b = []
        self.a = []
        for l0, l1 in zip(layers[:-1], layers[1:]):
            self.W.append(tf.Variable(tf.random.normal(shape=[l1[0], l0[0]], dtype=tf.dtypes.float64)))
            self.b.append(tf.Variable(tf.random.normal(shape=[l1[0], 1], dtype=tf.dtypes.float64)))
            self.a.append(l1[1])

    #---------------------------------------------------------------------------
    # Implementation of the forward pass, compute the network output y given the
    # input x.
    #
    # Input: x is a tensor of dimension dxN, with N the number of patterns
    #        and d the dimension (number of features).
    #
    # Return: the activation y in the last layer, which must be a tensor of
    #         dimension 1xN.
    #---------------------------------------------------------------------------
    def predict(self, x):
        #-----------------------------------------------------------------------
        # TO-DO block: loop in the network layers computing the activations.
        # The activation of the last layer should be stored at variable y to
        # be returned.
        #-----------------------------------------------------------------------
        pass
        #-----------------------------------------------------------------------
        # End of TO-DO block
        #-----------------------------------------------------------------------

        return y

    #---------------------------------------------------------------------------
    # Implementation of the backward pass, compute the gradients of the loss
    # function with respect to all the weights and biases using the TensorFlow
    # GradientTape.
    #
    # Note that the gradient tape must be persistent as many gradients must be
    # computed.
    #
    # Input: x is a tensor of dimension dxN, with N the number of patterns
    #        and d the dimension (number of features).
    #        t is a tensor of dimension 1xN, with N the number of patterns.
    #
    # Return: dW is a list with the gradients of the loss function with respect
    #         to the weights of each layer
    #         db is a list with the gradients of the loss function with respect
    #         to the biases of each layer
    #---------------------------------------------------------------------------
    def compute_gradients(self, x, t, loss_function):
        #-----------------------------------------------------------------------
        # TO-DO block: compute the gradients db, dW
        #-----------------------------------------------------------------------
        pass
        #-----------------------------------------------------------------------
        # End of TO-DO block
        #-----------------------------------------------------------------------

        return db, dW

    #---------------------------------------------------------------------------
    # Gradient step:
    #---------------------------------------------------------------------------
    def gradient_step(self, x, t, eta, loss_function):
        dB, dW = self.compute_gradients(x, t, loss_function)

        #-----------------------------------------------------------------------
        # TO-DO block: Loop in layers updating the model parameters b and w
        #-----------------------------------------------------------------------
        pass
        #-----------------------------------------------------------------------
        # End of TO-DO block
        #-----------------------------------------------------------------------

    #---------------------------------------------------------------------------
    # Fit the model parameters to the training data x, t.
    # Return the loss at each training epoch.
    #---------------------------------------------------------------------------
    def fit(self, x, t, eta, num_epochs, batch_size, loss_function):
        (dim, n) = x.shape
        num_batches = (n // batch_size) + ((n % batch_size) != 0)

        loss = np.zeros(num_epochs)
        for i in range(num_epochs):
            # Shuffle data and generate batches:
            ix = np.random.permutation(n)
            for j in range(num_batches):
                imin = j*batch_size
                imax = np.minimum((j+1)*batch_size, n)

                ibatch = ix[imin:imax]
                batch_x = x[:, ibatch]
                batch_t = t[:, ibatch]
                self.gradient_step(batch_x, batch_t, eta, loss_function)

            # Calculo el loss de la epoca con todos los datos:
            loss[i] = self.get_loss(x, t, loss_function).numpy()
        return loss

    #---------------------------------------------------------------------------
    # Compute loss:
    #---------------------------------------------------------------------------
    def get_loss(self, x, t, loss_function):
        y = self.predict(x)
        return loss_function(y, t)
    

    
