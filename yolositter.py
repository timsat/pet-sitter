import cv2
import uuid
import os
import numpy as np
import vlc
import glob
import random
import time
import logging
import handlers

confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image
CHECK_PERIOD = 3
frameNum = 0

def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def drawPred(frame, classes, classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255))

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs, classes):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(frame, classes, classIds[i], confidences[i], left, top, left + width, top + height)


def main():
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=logging.DEBUG)
    LOGGER = logging.getLogger(__name__)

    LOGGER.warning("test log record")


    # Load names of classes
    classesFile = "yolo/coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Give the configuration and weight files for the model and load the network using them.
    modelConfiguration = "yolo/yolov3.cfg"
    modelWeights = "yolo/yolov3.weights"

    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    last_checked = 0

    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False

    while rval:
        # vc.grab()
        rval, frame = vc.read(frame)
        key = cv2.waitKey(1)
        if key == 27:     # exit on ESC
            break

        if (time.time() - last_checked) > CHECK_PERIOD:
            last_checked = time.time()
            # Create a 4D blob from a frame.
            blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

            # Sets the input to the network
            net.setInput(blob)

            # Runs the forward pass to get output of the output layers
            outs = net.forward(getOutputsNames(net))

            global frameNum
            frameNum += 1
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]

            # Scan through all the bounding boxes output from the network and keep only the
            # ones with high confidence scores. Assign the box's class label as the class with the highest score.
            classIds = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    classId = np.argmax(scores)
                    confidence = scores[classId]
                    if confidence > confThreshold:
                        center_x = int(detection[0] * frameWidth)
                        center_y = int(detection[1] * frameHeight)
                        width = int(detection[2] * frameWidth)
                        height = int(detection[3] * frameHeight)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)
                        classIds.append(classId)
                        confidences.append(float(confidence))
                        boxes.append([left, top, width, height])

            # Perform non maximum suppression to eliminate redundant overlapping boxes with
            # lower confidences.
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)

            for i in np.unique(indices):
                handler = getattr(handlers, "handle_" + classes[classIds[i]], lambda x, y: None)
                handler(frame, frameNum)

        # Remove the bounding boxes with low confidence
        postprocess(frame, outs, classes)

        # Put efficiency information. The function getPerfProfile returns the
        # overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
        cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv2.imshow("preview", frame)

if __name__ == "__main__":
    main()
