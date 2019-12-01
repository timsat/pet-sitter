from utils import warn, save_img_with_prefix
import logging

LOGGER = logging.getLogger("handlers")
inRowFramesWithDog = 0
previousFrameWithDog = 0


def _handle_shanya(img, frame_num):
    global inRowFramesWithDog
    global previousFrameWithDog
    if previousFrameWithDog + 1 == frame_num:
        inRowFramesWithDog += 1
    else:
        inRowFramesWithDog = 1

    LOGGER.debug("Dog detected for %d times")

    previousFrameWithDog = frame_num

    if inRowFramesWithDog > 2:
        LOGGER.critical("Dog detected for 3 or more time in a row! Warn trespasser!")
        warn()
        save_img_with_prefix(img, "./shanya-")


def handle_shanya(img, frameNum):
    _handle_shanya(img, frameNum)


def handle_dog(img, frameNum):
    _handle_shanya(img, frameNum)
