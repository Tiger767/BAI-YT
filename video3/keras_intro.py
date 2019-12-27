from time import sleep

import numpy as np
import tensorflow as tf
from tensorflow import keras

# To view images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# check if using GPU
print(tf.test.gpu_device_name())

# Allow growth (No way yet for v2)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True   
sess = tf.compat.v1.Session(config=config)

# Get data
(tx, ty), (vx, vy) = keras.datasets.fashion_mnist.load_data()
labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 
          'Coat', 'Sandal', 'Shirt', 'Sneaker', 
          'Bag', 'Ankle boot']
print('Initial Training shapes', tx.shape, ty.shape)
ndx = np.random.choice(range(tx.shape[0]), size=10000, replace=False)
tx, ty = np.expand_dims(tx[ndx], axis=-1), ty[ndx]
ty = np.identity(len(labels))[ty]
print('Final Training shapes', tx.shape, ty.shape)
print('Initial Validation shapes', vx.shape, vy.shape)
vx = np.expand_dims(vx, axis=-1)
vy = np.identity(len(labels))[vy]
print('Final Validation shapes', vx.shape, vy.shape)

# Build Model functions
def basic_dense_model():
    inputs = keras.layers.Input(shape=tx.shape[1:])
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(len(labels), activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'], experimental_run_tf_function=False)
    return model

def dense_model():
    inputs = keras.layers.Input(shape=tx.shape[1:])
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=.99)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=.99)(x)
    outputs = keras.layers.Dense(len(labels), activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'], experimental_run_tf_function=True)
    return model

def basic_conv_model():
    inputs = keras.layers.Input(shape=tx.shape[1:])
    x = keras.layers.Conv2D(8, kernel_size=5, strides=2,
                            activation='relu')(inputs)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(len(labels), activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'], experimental_run_tf_function=False)
    return model

def conv_model():
    inputs = keras.layers.Input(shape=tx.shape[1:])
    x = keras.layers.Conv2D(8, kernel_size=5, strides=2, 
                            activation='relu')(inputs)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=.99)(x)
    x = keras.layers.Conv2D(16, kernel_size=3, strides=2,
                            activation='relu')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=.99)(x)
    x = keras.layers.Conv2D(32, kernel_size=3, strides=2,
                            activation='relu')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=.99)(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=.99)(x)
    outputs = keras.layers.Dense(len(labels), activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                metrics=['accuracy'], experimental_run_tf_function=False)
    return model

# Get Model
model_funcs = [basic_dense_model, dense_model, basic_conv_model, conv_model]
for model_func in model_funcs:
    name = model_func.__name__
    print(name)
    model = model_func()
    model.summary()

    # Train
    model.fit(tx, ty, validation_data=(vx, vy), batch_size=256, epochs=2, verbose=1)

    # Save weights
    # model.save_weights(f'{name}_weights.h5')

    # Predict
    num_ndxs = np.minimum(10, vx.shape[0])
    for ndx in range(num_ndxs):
        pred = model(np.expand_dims(vx[ndx], axis=0).astype(np.float32))[0]
        truth = labels[np.argmax(vy[ndx])]
        label = labels[np.argmax(pred)]
        print(f'{ndx+1}/{num_ndxs}. Truth: {truth} - Prediction: {label} - '
              f'Correct: {truth == label}')
        plt.imshow(np.squeeze(vx[ndx]))
        #plt.show()
        plt.pause(.1)
    plt.close()
    sleep(4)