import cv2
import uuid
import os
import numpy as np
import vlc
import glob
import random
import time
import logging

def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    LOGGER = logging.getLogger(__name__)

    confThreshold = 0.5  #Confidence threshold
    nmsThreshold = 0.4   #Non-maximum suppression threshold
    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image

    # Load names of classes
    classesFile = "coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "yolov3.cfg"
    modelWeights = "yolov3.weights"

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

