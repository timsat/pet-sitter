from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import argparse
import json
import functools
import os

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

CLASS_NAMES = np.array(['cat', 'dog'])
IMG_WIDTH = 300
IMG_HEIGHT = 300
BATCH_SIZE = 64


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='mse', metrics=['accuracy'])
    return model


def decode_img(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])


def process_path(labels_tensor, file_path):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    labels_joined = labels_tensor.lookup(file_path)
    labels = tf.strings.split(labels_joined, sep=',')
    return img, tf.dtypes.cast(labels == CLASS_NAMES, tf.float32)


def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10,10))
    for n in range(25):
        ax = plt.subplot(5,5,n+1)
        plt.imshow(image_batch[n])
        plt.title(str(label_batch[n]))
        plt.axis('off')
    plt.show()


def prepare_for_training(ds, cache=True):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.

    return ds


def load_dataset(filename, cache = None):
    ds_json = None
    with open(filename, 'r') as f:
        ds_json = json.load(f)

    ds_size = len(ds_json)



    list_ds = tf.data.Dataset.from_tensor_slices(list(ds_json.keys()))
    keys_tensor = tf.constant(list(ds_json.keys()))

    vals = []
    for key in ds_json.keys():
        vals.append(','.join(ds_json[key]))
    labels_tensor = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(keys_tensor, tf.constant(vals)), "not_found")
    ds = list_ds.map(functools.partial(process_path, labels_tensor), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.shuffle(buffer_size=ds_size)
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()


    return ds, labels_tensor, ds_size

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', '-d', type=str, required=True, help='dataset file')
args = parser.parse_args()

ds, labels_tensor, ds_size = load_dataset(args.dataset, cache="./mynet.tfcache")

train_size = int(0.85 * ds_size)
val_size = int(0.15 * ds_size)

val_ds = ds.skip(train_size).take(val_size)
train_ds = ds.take(train_size)
train_ds = train_ds.batch(train_size)
# Repeat forever
train_ds = train_ds.repeat()
# `prefetch` lets the dataset fetch batches in the background while the model
# is training.
train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

for image, label in ds.take(3):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())
    print("Label shape: ", label.shape)

train_iter = iter(train_ds)
imgs_batch, labels_batch = next(train_iter)
show_batch(imgs_batch, labels_batch)

model = create_model()

model.summary()

timgs_batch, tlabels_batch = next(iter(val_ds.batch(val_size)))
history = model.fit(imgs_batch, labels_batch, epochs=100, batch_size=64,
                    validation_data=(timgs_batch, tlabels_batch), callbacks=[cp_callback])
imgs_batch, labels_batch = next(train_iter)
show_batch(imgs_batch, labels_batch)
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

