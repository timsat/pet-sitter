import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers
import matplotlib.pylab as plt

import numpy as np
import PIL.Image as Image


IMAGE_SHAPE = (224, 224)


m = tf.keras.Sequential([
    hub.KerasLayer("https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4", trainable=True)
])
m.build([None, 224, 224, 3])  # Batch input shape.

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory("ourpets-photos", target_size=IMAGE_SHAPE)

for image_batch, label_batch in image_data:
    print("Image batch shape: ", image_batch.shape)
    print("Label batch shape: ", label_batch.shape)
    result_batch = m.predict(image_batch)
    print("Result batch shape: ", result_batch.shape)
    print("Result batch: ", result_batch)
    break


imagenet_labels = ['shanya', 'unknown']

