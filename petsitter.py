import cv2
import os
import tensorflow as tf
import numpy as np
import time
import logging
import handlers

from utils import save_img_with_prefix

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

MODEL_PATH = 'my-net'
CHECK_PERIOD = 2
COLLECT_PERIOD = 100


def predict(img, sess, tensor):
    #Loading the file
    #Format for the Mul:0 Tensor
    img2 = cv2.resize(img, dsize=(299, 299), interpolation=cv2.INTER_CUBIC)
    #Numpy array
    np_image_data = np.asarray(img2)
    np_image_data=cv2.normalize(np_image_data.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX)
    #maybe insert float convertion here - see edit remark!
    np_final = np.expand_dims(np_image_data, axis=0)
    return sess.run(tensor, {'Mul:0': np_final})


last_collected = 0

labels = [line.rstrip() for line in tf.gfile.GFile(os.path.join(MODEL_PATH, 'ourpets_labels.txt'))]

cv2.namedWindow("preview")
vc = cv2.VideoCapture(2)


def check():
    global last_collected
    p = list(zip(predict(frame, sess, tensor).tolist()[0], labels))
    t = time.time()
    detected = max(p, key=lambda x: x[0])
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.info("%s detected %s", detected[1], p)
    else:
        LOGGER.info("%s detected", detected[1])
    handler = getattr(handlers, "handle_" + detected[1], lambda x: None)
    handler(frame)
    if (t - last_collected) > COLLECT_PERIOD:
        LOGGER.info("periodic collect...")
        last_collected = t
        save_img_with_prefix(frame, detected[1] + "-")


with tf.gfile.FastGFile(os.path.join(MODEL_PATH, 'ourpets.pb'), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    tensor = sess.graph.get_tensor_by_name('final_result:0')

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    last_checked = 0
    last_collected = 0
    while rval:
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:     # exit on ESC
            break
        elif key == 99:   # c key
            LOGGER.info("s key pressed, saving image labeled as catty")
            save_img_with_prefix(frame, "./catty-")
        elif key == 115:  # s key
            LOGGER.info("s key pressed, saving image labeled as shanya")
            save_img_with_prefix(frame, "./shanya-")
        elif key == 117:  # u key
            LOGGER.info("u key pressed, saving image labeled as unknown")
            save_img_with_prefix(frame, "./unknown-")
        elif key == 112:  # p key
            LOGGER.info("p key pressed, running predictions")
            check()

        if (time.time() - last_checked) > CHECK_PERIOD:
            last_checked = time.time()
            check()


cv2.destroyWindow("preview")
