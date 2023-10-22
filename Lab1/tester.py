import numpy as np
import tensorflow as tf
from dlfbt_lab1 import NeuralNetwork
import pickle
import os 
import dlfbt

if __name__ == '__main__':
    os.chdir('FundamentosDeepLearning/Practicas/FundamentosDL/Lab1')

    np.random.seed(17)
    x = np.random.randn(3, 20)
    t = np.random.randn(1, 20)

    net = NeuralNetwork([(3, 'input'), (10, 'sigmoid'), (1, 'linear')])
    dW, db = net.compute_gradients(x, t)

    assert dW[0].shape == (10, 3)
    assert db[0].shape == (10, 1)
    assert dW[1].shape == (1, 10)
    assert db[1].shape == (1, 1)

    # Array values, should match those on the test file:
    with open('./test_nn_numpy_compute_gradients.pickle', 'rb') as handle:
        [dWexp, dbexp] = pickle.load(handle)

    tol = 1.e-8
    for dwp, dbp, dwe, dbe in zip(dW, db, dWexp, dbexp):
        assert np.max(np.abs(dwp - dwe)) < tol
        assert np.max(np.abs(dbp - dbe)) < tol

    dg = dlfbt.DataGeneratorLinear(a=[2.0, 2.0])
    dg.create_dataset(n=1000, seed=17, noise=1.0)
    x = dg.x.transpose()
    t = dg.t.transpose()

    np.random.seed(23)
    net = NeuralNetwork([(2, 'input'), (1, 'linear')])

    dW, db = net.compute_gradients(x, t)
    
    
    loss = net.fit(x, t, 0.01, 100, 1000, NeuralNetwork.mse_loss)
    z, y = net.predict(x)

    assert z[0].shape == (1, 1000)
    assert y[0].shape == (1, 1000)

    # Array values, should match those on the test file:
    with open('./test_nn_numpy_fit.pickle', 'rb') as handle:
        [zexp, yexp] = pickle.load(handle)

    tol = 1.e-8
    for zp, yp, ze, ye in zip(z, y, zexp, yexp):
        assert np.max(np.abs(zp - ze)) < tol
        assert np.max(np.abs(yp - ye)) < tol
    