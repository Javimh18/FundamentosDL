import pandas as pd
# Imports
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

import keras as k
from keras.models import Sequential
from keras.layers import Dense

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns; sns.set()

def plot_history_and_metrics(model:Sequential, history, x, y):
    # Plot history
    print(history.history.keys())

    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Accuracy')
    plt.ylabel('')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Loss per epoch')
    plt.ylabel('')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()

    # Evaluate (similar to fit but just 1 epoch iteration without changing the network)
    loss, accuracy = model.evaluate(x, y)
    print('Accuracy: %.2f' % (accuracy*100))