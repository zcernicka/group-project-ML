# If error [No module named 'sklearn'], in terminal: conda install -c conda-forge scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
#matplotlib notebook
#from matplotlib import pyplot as plt
#import seaborn as sb
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
# pip install git+https://github.com/tensorflow/docs
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
# Libraries and options
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten
from sklearn.model_selection import train_test_split

class neural_network(object):

    def __init__(self):
        pass

    def initial_sequence(self, df):

        x = df.loc[:, df.columns != 'price_sqm2']
        y = df['price_sqm2']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=9)

        # Initiate a sequential model (i.e., no recurrence)
        model = Sequential()

        # Make the first layer
        model.add(Dense(128, kernel_initializer='normal', input_dim=x_train.shape[1], activation='relu'))

        # Make hidden layers
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))
        model.add(Dense(256, kernel_initializer='normal', activation='relu'))

        # Make the output layer
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))

        # Compile the network
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
        model.summary()

        # Define how to name the files
        checkpoint_name = 'Weights-{epoch:03d}--{val_loss:.5f}.hdf5'
        # Instantiate the checkpoint system
        checkpoint = ModelCheckpoint(checkpoint_name, monitor='val_loss', verbose=2, save_best_only=True, mode='auto')
        callbacks_list = [checkpoint]

        early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)
        history = model.fit(x_train, y_train, epochs=100,
                               # No. of randomly sampled data points used to compute the errors at each epoch (avoid overfitting)
                               batch_size=32,
                               # Size of validation set for cross-validation
                               validation_split=0.2,
                               # Link to checkpoint system, to check teh best model previously built
                               callbacks=([early_stopping]))

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        return print(hist.tail()), model