import cv2
import numpy as np
import time
import logging
import handlers
import mynet

from utils import save_img_with_prefix

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
LOGGER = logging.getLogger(__name__)

CHECK_PERIOD = 2
COLLECT_PERIOD = 100


def predict(img, model):
    img2 = cv2.resize(img, dsize=(mynet.IMG_WIDTH, mynet.IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
    np_image_data = np.asarray(img2)
    np_image_data = cv2.normalize(np_image_data.astype('float'), None, 0, 1, cv2.NORM_MINMAX)
    #maybe insert float convertion here - see edit remark!
    np_image_data = np.expand_dims(np_image_data, axis=0)
    return model.predict(np_image_data)


last_collected = 0

labels = mynet.CLASS_NAMES

cv2.namedWindow("preview")
vc = cv2.VideoCapture(2)


def check(model):
    global last_collected
    global LOGGER
    p = list(zip(predict(frame, model).tolist()[0], labels))
    t = time.time()
    detected = filter(lambda x: x[0] > 0.5, p)
    if LOGGER.isEnabledFor(logging.DEBUG):
        LOGGER.info("%s detected %s", str(detected), str(p))
    else:
        LOGGER.info("%s detected", str(detected))
    #handler = getattr(handlers, "handle_" + detected[1], lambda x: None)
    #handler(frame)
    if (t - last_collected) > COLLECT_PERIOD:
        LOGGER.info("periodic collect...")
        last_collected = t
        save_img_with_prefix(frame, "-")


model = mynet.create_model()
mynet.load_weights(model)

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
        check(model)

    if (time.time() - last_checked) > CHECK_PERIOD:
        last_checked = time.time()
        check(model)


cv2.destroyWindow("preview")
