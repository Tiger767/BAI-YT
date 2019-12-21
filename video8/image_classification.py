import os
import numpy as np
import tensorflow as tf
from time import time

import image as img
from neural_network import (
    Trainner, Predictor, dense,
    conv2d
)
from util_funcs import (
    load_directory_dataset, save_h5py, load_h5py
)
from analytics import Analyzer


#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth = True
# config.gpu_options.force_gpu_compatible = True
#session = tf.compat.v1.Session(config=config)

TRAINING = False
PATH = 'trained_weights'
assert PATH is not None or TRAINING, (
    'Must provided a path if not training.'
)
DATA_MULTIPLIER = 5
EPOCHS = 25
BATCH_SIZE = 64
TARGET_SHAPE = (64, 64)
COLOR = False
DATA_PATH = '101_ObjectCategories'
FOLDERS = ['BACKGROUND_Google', 'Faces', 'Faces_easy', 'Leopards',
           'Motorbikes', 'accordion', 'airplanes', 'anchor', 'ant',
           'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain',
           'brontosaurus', 'buddha', 'butterfly', 'camera', 'cannon',
           'car_side', 'ceiling_fan', 'cellphone', 'chair', 'chandelier',
           'cougar_body', 'cougar_face', 'crab', 'crayfish', 'crocodile',
           'crocodile_head', 'cup', 'dalmatian', 'dollar_bill', 'dolphin',
           'dragonfly', 'electric_guitar', 'elephant', 'emu', 'euphonium',
           'ewer', 'ferry', 'flamingo', 'flamingo_head', 'garfield', 'gerenuk',
           'gramophone', 'grand_piano', 'hawksbill', 'headphone', 'hedgehog',
           'helicopter', 'ibis', 'inline_skate', 'joshua_tree', 'kangaroo',
           'ketch', 'lamp', 'laptop', 'llama', 'lobster', 'lotus', 'mandolin',
           'mayfly', 'menorah', 'metronome', 'minaret', 'nautilus', 'octopus',
           'okapi', 'pagoda', 'panda', 'pigeon', 'pizza', 'platypus',
           'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone',
           'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy',
           'soccer_ball', 'stapler', 'starfish', 'stegosaurus', 'stop_sign',
           'strawberry', 'sunflower', 'tick', 'trilobite', 'umbrella',
           'watch', 'water_lilly', 'wheelchair', 'wild_cat', 'windsor_chair',
           'wrench', 'yin_yang']
WS = img.Windows()

# Create Data Set if it does not exist
if not os.path.exists(DATA_PATH):
    raise Exception('Download Dataset: http://www.vision.caltech.edu/'
                    'Image_Datasets/Caltech101/Caltech101.html#Download')
if not os.path.exists('data.h5'):
    def load_x(path):
        return img.load(path, target_shape=TARGET_SHAPE, color=COLOR)

    dataset = load_directory_dataset(DATA_PATH, load_x)

    a = Analyzer(dataset['train_x'], dataset['train_y'], FOLDERS)
    print(a.calculate_distribution_of_labels())
    input('Enter to continue')

    w1 = WS.add('Original Image')
    w2 = WS.add('Mutated Image 1')
    w3 = WS.add('Mutated Image 2')
    WS.start()

    xs = []
    ys = []
    test_x = []
    test_y = []
    for x, y in zip(dataset['train_x'], dataset['train_y']):
        WS.set(w1, x)

        for _ in range(DATA_MULTIPLIER):
            mutated = img.set_gamma(x, np.random.uniform(.5, 1.5))
            mutated = img.rotate(x, np.random.randint(-15, 16))
            mutated = img.translate(x, np.random.randint(-10, 11),
                                    np.random.randint(-10, 11))
            xs.append(mutated)
            ys.append(y)
            WS.set(w2, mutated)

        mutated = img.set_gamma(x, np.random.uniform(.5, 1.5))
        mutated = img.rotate(x, np.random.randint(-15, 16))
        mutated = img.translate(x, np.random.randint(-10, 11),
                                np.random.randint(-10, 11))
        test_x.append(mutated)
        test_y.append(y)
        WS.set(w3, mutated)

    WS.stop()


    dataset['train_x'] = np.concatenate([dataset['train_x'], xs], axis=0)
    dataset['train_y'] = np.concatenate([dataset['train_y'], ys], axis=0)

    if COLOR:
        dataset['test_x'] = np.array(test_x)
    else:
        dataset['train_x'] = np.expand_dims(dataset['train_x'], axis=-1)
        dataset['test_x'] = np.expand_dims(test_x, axis=-1)

    # Makes data.h5 really large, but can improve accuracy
    # dataset['train_x'] = (dataset['train_x'] - 127.5) / 127.5
    # dataset['test_x'] = (dataset['test_x'] - 127.5) / 127.5

    dataset['test_y'] = np.array(test_y)
    save_h5py('data.h5', dataset)
    del dataset, xs, test_x, ys, test_y

# Create model
path = PATH
if TRAINING:
    if COLOR:
        shape = (*TARGET_SHAPE, 3)
    else:
        shape = (*TARGET_SHAPE, 1)
    inputs = tf.keras.layers.Input(shape=shape)
    x = conv2d(64, 3, 2)(inputs)
    x = conv2d(128, 3, 2)(x)
    x = conv2d(256, 3, 2)(x)
    x = conv2d(512, 3, 2)(x)
    x = conv2d(1024, 3, 2)(x)
    x = conv2d(2048, 3, 2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = dense(512)(x)
    outputs = dense(len(FOLDERS), activation='softmax', batch_norm=False)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(.001, amsgrad=True),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    trainner = Trainner(model, 'data.h5')
    if PATH is not None:
        trainner.load(PATH,
                      tf.keras.optimizers.Adam(.001, amsgrad=True),
                      'categorical_crossentropy',
                      ['accuracy'])
    s = time()
    trainner.train(EPOCHS, batch_size=BATCH_SIZE, verbose=True)
    print(f'Full Training Duration: {time()-s} seconds')

    path = trainner.save('')

predictor = Predictor(path)
w = WS.add('Image')
WS.start()
dataset = load_h5py('data.h5')
test_x, test_y = dataset['test_x'], dataset['test_y']
for ndx in range(test_x.shape[0]):
    WS.set(w, np.squeeze(test_x[ndx]))
    pred = predictor.predict(test_x[ndx])
    print(f'Prediction: {pred.round(3)}\n'
          f'Class Predicted: {FOLDERS[np.argmax(pred)]}\n'
          f'True Class: {FOLDERS[np.argmax(test_y[ndx])]}')
    input('Enter for next image')
WS.stop()