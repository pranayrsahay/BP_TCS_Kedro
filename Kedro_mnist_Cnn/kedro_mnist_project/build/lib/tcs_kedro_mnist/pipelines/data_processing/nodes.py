"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.10
"""
"""
The pipeline loads the MNIST data and does the normalizing and re-shaping of data in the preprocessing node
"""
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)

#Loading MNIST data using Keras dataset and saves it in the data/01_raw for future use as a numpy custom dataset
def load_mnist_data():
    logger.info("Loading MNIST data")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_train, y_train, x_test, y_test

# Normalizing and reshaping of Train and Test Data
def preprocess(x_train, x_test):
    logger.info("Preprocessing(Normalizing) MNIST data")
    x_train_normalized = x_train.reshape((x_train.shape[0], 28, 28, 1)).astype('float32') / 255
    x_test_normalized = x_test.reshape((x_test.shape[0], 28, 28, 1)).astype('float32') / 255
    return x_train_normalized, x_test_normalized
