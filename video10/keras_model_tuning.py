import numpy as np
import tensorflow as tf
from tensorflow import keras
import evolution_algorithm as evoalgo


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True   
sess = tf.compat.v1.Session(config=config)

(tx, ty), (vx, vy) = keras.datasets.fashion_mnist.load_data()
labels = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 
          'Coat', 'Sandal', 'Shirt', 'Sneaker', 
          'Bag', 'Ankle boot']
print(tx.shape, vx.shape)
ndx = np.random.choice(range(tx.shape[0]), size=10000, replace=False)
tx, ty = np.expand_dims(tx[ndx], axis=-1), ty[ndx]
ty = np.identity(len(labels))[ty]
vx = np.expand_dims(vx, axis=-1)
vy = np.identity(len(labels))[vy]


hpt = evoalgo.HyperParameterTuner()
num_layers = hpt.uniform(1, 3, volatility=.3, integer=True)
filters1 = hpt.list([4, 8, 16, 32, 64], volatility=.3)
filters2 = hpt.list([4, 8, 16, 32, 64], volatility=.3)
filters3 = hpt.list([4, 8, 16, 32, 64], volatility=.3)
units1 = hpt.list([4, 8, 16, 32, 64, 128], volatility=.3)
lr = hpt.list([.1, .01, .001, .0001], volatility=.2)
batch_size = hpt.list([4, 8, 16, 32, 64, 128, 256], volatility=.2)

def eval_func():
    inputs = keras.layers.Input(shape=tx.shape[1:])

    x = keras.layers.Conv2D(filters1(), kernel_size=5, strides=2, 
                            activation='relu')(inputs)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=.99)(x)

    if num_layers() > 1:
        x = keras.layers.Conv2D(filters2(), kernel_size=3, strides=2,
                                activation='relu')(x)
        x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=.99)(x)
    if num_layers() > 2:
        x = keras.layers.Conv2D(filters3(), kernel_size=3, strides=2,
                                activation='relu')(x)
        x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=.99)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units1(), activation='relu')(x)
    x = keras.layers.BatchNormalization(epsilon=1e-5, momentum=.99)(x)

    outputs = keras.layers.Dense(len(labels), activation='softmax')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(tx, ty, validation_data=(vx, vy),
                        batch_size=batch_size(), epochs=5,
                        verbose=0).history
    print(history['loss'][-1], history['val_loss'][-1])
    print(num_layers(), filters1(), filters2(), filters3(),
          units1(), lr(), batch_size())
    return (history['loss'][-1] + history['val_loss'][-1]) / 2

hpt.tune(10, 10, 3, eval_func, verbose=True)

print(num_layers(), filters1(), filters2(), filters3(),
      units1(), lr(), batch_size())

# 1 32 32 64 32 .001 64