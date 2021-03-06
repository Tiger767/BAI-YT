"""
Author: Travis Hammond
Version: 12_21_2019
"""


import os
import datetime
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import model_from_json
from tensorflow.keras.regularizers import l1_l2

try:
    from utils.util_funcs import (
        load_directory_dataset, load_h5py, save_h5py
    )
except ImportError:
    from util_funcs import (
        load_directory_dataset, load_h5py, save_h5py
    )


class Trainner:
    """Trainner is used for loading, saving, and training keras models."""

    def __init__(self, model, data, file_loader=None, file_loader_y=None):
        """Initializes train, validation, and test data.
        params:
            model: A compiled keras model
            data: A dictionary or string (path) containg train data, and
                  optionally validation and test data.
                  Ex. {'train_x': [...], 'train_y: [...]}
            file_loader: A function for loading each X file
            file_loader_y: A function for loading each Y file
        """
        assert isinstance(data, (str, dict)), (
            'data must be either in a dictionary or a file/folder path'
        )
        self.model = model
        self.train_data = None
        self.validation_data = None
        self.test_data = None

        if isinstance(data, str):
            if os.path.isdir(data):
                assert file_loader is not None
                data = load_directory_dataset(data, file_loader,
                                              file_loader_y)
            else:
                assert data.split('.')[1] == 'h5'
                data = load_h5py(data)
        if isinstance(data, dict):
            if 'train_x' in data and 'train_y' in data:
                self.train_data = (np.array(data['train_x']),
                                   np.array(data['train_y']))
            else:
                raise Exception('There must be a train dataset')
            if 'validation_x' in data and 'validation_y' in data:
                self.validation_data = (np.array(data['validation_x']),
                                        np.array(data['validation_y']))
            if 'test_x' in data and 'test_y' in data:
                self.test_data = (np.array(data['test_x']),
                                  np.array(data['test_y']))
        else:
            raise ValueError('Invalid data')

    def train(self, epochs, batch_size=None, callbacks=None, verbose=True):
        """Trains the keras model.
        params:
            epochs: An integer, which is the number of complete
                    iterations to train
            batch_size: An integer, which is the number of samples
                        per graident update
            callbacks: A list of keras Callback instances,
                       which are called during training and validation
            verbose: A boolean, which determines the verbositiy level
        """
        self.model.fit(self.train_data[0], self.train_data[1],
                       validation_data=self.validation_data,
                       batch_size=batch_size, epochs=epochs,
                       verbose=1 if verbose else 0, callbacks=callbacks)
        if verbose:
            print('Train Data:')
            print(self.model.evaluate(self.train_data[0], self.train_data[1],
                                      batch_size=batch_size))
            if self.validation_data is not None:
                print('Validation Data:')
                print(self.model.evaluate(self.validation_data[0],
                                          self.validation_data[1],
                                          batch_size=batch_size))
            if self.test_data is not None:
                print('Test Data:')
                print(self.model.evaluate(self.test_data[0],
                                          self.test_data[1],
                                          batch_size=batch_size))

    def load(self, path, optimizer, loss, metrics=None):
        """Loads a model and weights from a file.
           (overrides the inital provided model)
        params:
            path: A string, which is the path to a folder
                  containing model.json, weights.h5, and note.txt
            optimizer: A string or optimizer instance, which will be
                       the optimizer for the loaded model
            loss: A string or loss instance, which will be
                  the loss function for the loaded model
            metrics: A list of metrics, which will be used
                     by the loaded model
        """
        with open(os.path.join(path, 'model.json'), 'r') as file:
            self.model = model_from_json(file.read())
            self.model.compile(optimizer=optimizer, loss=loss,
                               metrics=metrics)
        self.model.load_weights(os.path.join(path, 'weights.h5'))
        with open(os.path.join(path, 'note.txt'), 'r') as file:
            print(file.read(), end='')

    def save(self, path, note=None):
        """Saves the model and weights to a file.
        params:
            path: A string, which is the path to create a folder in
                  containing model.json, weights.h5, and note.txt
        return: A string, which is the given path + folder name
        """
        time = datetime.datetime.now()
        path = os.path.join(path, time.strftime(r'%Y%m%d_%H%M%S_%f'))
        os.mkdir(path)
        self.model.save_weights(os.path.join(path, 'weights.h5'))
        with open(os.path.join(path, 'model.json'), 'w') as file:
            file.write(self.model.to_json())
        with open(os.path.join(path, 'note.txt'), 'w') as file:
            if note is None:
                self.model.summary(print_fn=lambda line: file.write(line+'\n'))
            else:
                file.write(note)
        return path


class Predictor:
    """Predictor is used for loading and predicting keras models."""

    def __init__(self, path,  weights_name='weights.h5',
                 model_name='model.json'):
        """Initializes the model and weights.
        params:
            path: A string, which is the path to a folder containing
                  model.json, weights.h5, and maybe note.txt
            weights_name: A string, which is the name of the weights to load
            model_name: A string, which is the name of the model to laod
        """
        with open(os.path.join(path, model_name), 'r') as file:
            self.model = model_from_json(file.read())
        self.model.load_weights(os.path.join(path, weights_name))
        note_path = os.path.join(path, 'note.txt')
        if os.path.exists(note_path):
            with open(note_path, 'r') as file:
                print(file.read(), end='')

    def predict(self, x):
        """Predicts on a single sample.
        params:
            x: A single model input
        return: A result from the model output
        """
        return self.model.predict(np.expand_dims(x, axis=0))[0]

    def predict_all(self, x, batch_size=None):
        """Predicts on many samples.
        params:
            x: A ndarray of model inputs
        return: A result from the model output
        """
        return self.model.predict(x, batch_size=batch_size)


def dense(units, activation='relu', l1=0, l2=0, batch_norm=True,
          momentum=0.999, epsilon=1e-5, name=None):
    """Creates a dense layer function.
    params:
        units: An integer, which is the dimensionality of the output space
        activation: A string or keras/TF activation function
        l1: A float, which is the amount of L1 regularization
        l2: A float, which is the amount of L2 regularization
        batch_norm: A boolean, which determines if batch
                    normalization is enabled
        momentum: A float, which is the momentum for the moving
                  mean and variance
        epsilon: A float, which adds variance to avoid dividing by zero
        name: A string, which is the name of the dense layer
    return: A function, which takes a layer as input and returns a dense(layer)
    """
    if activation == 'relu':
        kernel_initializer = 'he_normal'
    else:
        kernel_initializer = 'glorot_uniform'
    if l1 == l2 == 0:
        dl = keras.layers.Dense(units, activation=activation, name=name,
                                kernel_initializer=kernel_initializer,
                                use_bias=not batch_norm)
    else:
        dl = keras.layers.Dense(units, activation=activation,
                                kernel_regularizer=l1_l2(l1, l2), name=name,
                                kernel_initializer=kernel_initializer,
                                use_bias=not batch_norm)
    if batch_norm:
        bn_name = name + '_batchnorm' if name is not None else None
        bnl = keras.layers.BatchNormalization(epsilon=epsilon,
                                              momentum=momentum,
                                              name=bn_name)

    def layer(x):
        """Applies dense layer to input layer.
        params:
            x: A Tensor
        return: A Tensor
        """
        x = dl(x)
        if batch_norm:
            x = bnl(x)
        return x
    return layer


def conv1d(filters, kernel_size, strides=1, activation='relu',
           padding='same', max_pool_size=None, max_pool_strides=None,
           upsampling_size=None, l1=0, l2=0, batch_norm=True,
           momentum=0.999, epsilon=1e-5, name=None):
    """Creates a 1D convolution layer function.
    params:
        filters: An integer, which is the dimensionality of the output space
        kernel_size: An integer or tuple of 1 integers, which is the size of
                     the convoluition kernel
        strides: An integer or tuple of 1 integers, which is stride length
                 of the windows
        activation: A string or keras/TF activation function
        padding: A string ('same', 'valid')
        max_pool_size: An integer, which is the size of the pooling windows
        max_pool_strides: An integer, which is the factor to downscale by
        upsampling_size: An integer, which is the factor to upsample by
        l1: A float, which is the amount of L1 regularization
        l2: A float, which is the amount of L2 regularization
        batch_norm: A boolean, which determines if batch
                    normalization is enabled
        momentum: A float, which is the momentum for the moving
                  mean and variance
        epsilon: A float, which adds variance to avoid dividing by zero
        name: A string, which is the name of the dense layer
    return: A function, which takes a layer as input and returns
            a conv1d(layer)
    """
    if activation == 'relu':
        kernel_initializer = 'he_normal'
    else:
        kernel_initializer = 'glorot_uniform'
    if l1 == l2 == 0:
        cl = keras.layers.Conv1D(filters, kernel_size,
                                    activation=activation,
                                    strides=strides, padding=padding,
                                    name=name, use_bias=not batch_norm,
                                    kernel_initializer=kernel_initializer)
    else:
        cl = keras.layers.Conv1D(filters, kernel_size,
                                    activation=activation,
                                    strides=strides, padding=padding,
                                    kernel_regularizer=l1_l2(l1, l2),
                                    name=name, use_bias=not batch_norm,
                                    kernel_initializer=kernel_initializer)
    if batch_norm:
        bn_name = name + '_batchnorm' if name is not None else None
        bnl = keras.layers.BatchNormalization(epsilon=epsilon,
                                                momentum=momentum,
                                                name=bn_name)
    if (max_pool_size is not None or max_pool_strides is not None):
        mp_name = name + '_maxpool' if name is not None else None
        mpl = keras.layers.MaxPooling1D(pool_size=max_pool_size,
                                        strides=max_pool_strides,
                                        name=mp_name)
    if upsampling_size is not None:
        us_name = name + '_upsample' if name is not None else None
        usl = keras.layers.UpSampling1D(upsampling_size, name=us_name)(x)

    def layer(x):
        """Applies 1D convolution layer to layer x.
        params:
            x: A Tensor
        return: A Tensor
        """
        x = cl(x)
        if batch_norm:
            x = bnl(x)
        if (max_pool_size is not None or max_pool_strides is not None):
            x = mpl(x)
        if upsampling_size is not None:
            x = usl(x)
        return x
    return layer


def conv2d(filters, kernel_size=3, strides=1, activation='relu',
           padding='same', max_pool_size=None, max_pool_strides=None,
           l1=0, l2=0, batch_norm=True, momentum=0.999, epsilon=1e-5,
           upsampling_size=None, transpose=False, name=None):
    """Creates a 2D convolution layer function.
    params:
        filters: An integer, which is the dimensionality of the output space
        kernel_size: An integer or tuple of 2 integers, which is the size of
                     the convoluition kernel
        strides: An integer or tuple of 2 integers, which is stride length
                 of the windows
        activation: A string or keras/TF activation function
        padding: A string ('same', 'valid')
        max_pool_size: An integer or tuple of 2 integers, which is the size
                       of the pooling windows
        max_pool_strides: An integer or tuple of 2 integers, which is the
                          factor to downscale by
        l1: A float, which is the amount of L1 regularization
        l2: A float, which is the amount of L2 regularization
        batch_norm: A boolean, which determines if batch
                    normalization is enabled
        momentum: A float, which is the momentum for the moving
                  mean and variance
        epsilon: A float, which adds variance to avoid dividing by zero
        upsampling_size: An integer, which is the factor to upsample by
        transpose: A boolean, which determines if the convolution layer
                   should be a deconvolution layer
        name: A string, which is the name of the dense layer
    return: A function, which takes a layer as input and returns
            a conv2d(layer)
    """
    if activation == 'relu':
        kernel_initializer = 'he_normal'
    else:
        kernel_initializer = 'glorot_uniform'
    if transpose:
        kl_conv2d = keras.layers.Conv2DTranspose
    else:
        kl_conv2d = keras.layers.Conv2D
    if l1 == l2 == 0:
        cl = kl_conv2d(filters, kernel_size,
                       activation=activation,
                       strides=strides, padding=padding,
                       name=name, use_bias=not batch_norm,
                       kernel_initializer=kernel_initializer)
    else:
        cl = kl_conv2d(filters, kernel_size,
                       activation=activation,
                       strides=strides, padding=padding,
                       kernel_regularizer=l1_l2(l1, l2),
                       name=name, use_bias=not batch_norm,
                       kernel_initializer=kernel_initializer)
    if batch_norm:
        bn_name = name + '_batchnorm' if name is not None else None
        bnl = keras.layers.BatchNormalization(epsilon=epsilon,
                                              momentum=momentum,
                                              name=bn_name)
    if (max_pool_size is not None or max_pool_strides is not None):
        mp_name = name + '_maxpool' if name is not None else None
        mpl = keras.layers.MaxPooling2D(pool_size=max_pool_size,
                                        strides=max_pool_strides,
                                        name=mp_name)
    if upsampling_size is not None:
        us_name = name + '_upsample' if name is not None else None
        usl = keras.layers.UpSampling2D(upsampling_size, name=us_name)

    def layer(x):
        """Applies 2D convolution layer to layer x.
        params:
            x: A Tensor
        return: A Tensor
        """
        x = cl(x)
        if batch_norm:
            x = bnl(x)
        if (max_pool_size is not None or max_pool_strides is not None):
            x = mpl(x)
        if upsampling_size is not None:
            x = usl(x)
        return x
    return layer


def inception(inceptions):
    """Creates an inception network.
    params:
        inceptions: A list of functions that apply layers or Tensors
    return: A function, which takes a layer and returns inception(layer)
    """
    def layer(x):
        """Builds and applies an inception architecture.
        params:
            x: A Tensor
        return: A Tensor
        """
        branches = []
        for branch in inception:
            y = branch[0](x)
            for layer in branch[1:]:
                y = layer(y)
            branches.append(y)
        return branches
    return layer


if __name__ == '__main__':
    tx = np.array([[0, 0], [1, 1]])
    ty = np.array([0, 1])
    save_h5py('data.h5', {'train_x': tx, 'train_y': ty})

    inputs = keras.layers.Input(shape=(2,))
    x = dense(16)(inputs)
    outputs = dense(1, activation='sigmoid', batch_norm=False)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])

    trainner = Trainner(model, 'data.h5')
    trainner.train(10000)
    path = trainner.save('')
    predictor = Predictor(path)
    print(predictor.predict([0, 1]))
