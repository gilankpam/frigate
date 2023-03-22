import numpy as np
import re
import math
import random
import cv2

import logging
from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from typing import Literal
from pydantic import Field

logger = logging.getLogger(__name__)

DETECTOR_KEY = "rknn"

INPUT_SIZE = 300

NUM_RESULTS = 1917
NUM_CLASSES = 91

Y_SCALE = 10.0
X_SCALE = 10.0
H_SCALE = 5.0
W_SCALE = 5.0

def expit(x):
    return 1. / (1. + math.exp(-x))


def unexpit(y):
    return -1.0 * math.log((1.0 / y) - 1.0)


def CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1):
    w = max(0.0, min(xmax0, xmax1) - max(xmin0, xmin1))
    h = max(0.0, min(ymax0, ymax1) - max(ymin0, ymin1))
    i = w * h
    u = (xmax0 - xmin0) * (ymax0 - ymin0) + (xmax1 - xmin1) * (ymax1 - ymin1) - i

    if u <= 0.0:
        return 0.0

    return i / u

def load_box_priors(path):
    box_priors_ = []
    fp = open(path, 'r')
    ls = fp.readlines()
    for s in ls:
        aList = re.findall('([-+]?\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', s)
        for ss in aList:
            aNum = float((ss[0]+ss[2]))
            box_priors_.append(aNum)
    fp.close()

    box_priors = np.array(box_priors_)
    box_priors = box_priors.reshape(4, NUM_RESULTS)

    return box_priors

class RknnDetectorConfig(BaseDetectorConfig):
    type: Literal[DETECTOR_KEY]
    """
    auto = 0
    core 0 = 1
    core 1 = 2
    core 2 = 4
    core 1 & 2 = 3
    core 1 & 2 & 3 = 7
    """
    core_mask: int = Field(default=0, title="Set NPU working core mode")

class Rknn(DetectionApi):
    type_key = DETECTOR_KEY

    def __init__(self, config: RknnDetectorConfig):
        from rknnlite.api import RKNNLite
        self.rknn_lite = RKNNLite()
        if self.rknn_lite.load_rknn(config.path) != 0:
            logger.error('Error initialize rknn model')
        if self.rknn_lite.init_runtime(core_mask=config.core_mask) != 0:
            logger.error('Error init rknn runtime')
        self.box_priors = load_box_priors('/box_priors.txt')

    def detect_raw(self, tensor_input):
        outputs = self.rknn_lite.inference(inputs=[tensor_input])
        predictions = outputs[0].reshape((1, NUM_RESULTS, 4))
        outputClasses = outputs[1].reshape((1, NUM_RESULTS, NUM_CLASSES))
        candidateBox = np.zeros([2, NUM_RESULTS], dtype=int)
        classScore = [-1000.0] * NUM_RESULTS
        vaildCnt = 0

        box_priors = self.box_priors

        # Post Process
        # got valid candidate box
        for i in range(0, NUM_RESULTS):
            topClassScoreIndex = np.argmax(outputClasses[0][i][1:]) + 1
            topClassScore = expit(outputClasses[0][i][topClassScoreIndex])
            if topClassScore > 0.4:
                candidateBox[0][vaildCnt] = i
                candidateBox[1][vaildCnt] = topClassScoreIndex
                classScore[vaildCnt] = topClassScore
                vaildCnt += 1

        # calc position
        for i in range(0, vaildCnt):
            if candidateBox[0][i] == -1:
                continue

            n = candidateBox[0][i]
            ycenter = predictions[0][n][0] / Y_SCALE * box_priors[2][n] + box_priors[0][n]
            xcenter = predictions[0][n][1] / X_SCALE * box_priors[3][n] + box_priors[1][n]
            h = math.exp(predictions[0][n][2] / H_SCALE) * box_priors[2][n]
            w = math.exp(predictions[0][n][3] / W_SCALE) * box_priors[3][n]

            ymin = ycenter - h / 2.
            xmin = xcenter - w / 2.
            ymax = ycenter + h / 2.
            xmax = xcenter + w / 2.

            predictions[0][n][0] = ymin
            predictions[0][n][1] = xmin
            predictions[0][n][2] = ymax
            predictions[0][n][3] = xmax
        
        # NMS
        for i in range(0, vaildCnt):
            if candidateBox[0][i] == -1:
                continue

            n = candidateBox[0][i]
            xmin0 = predictions[0][n][1]
            ymin0 = predictions[0][n][0]
            xmax0 = predictions[0][n][3]
            ymax0 = predictions[0][n][2]

            for j in range(i+1, vaildCnt):
                m = candidateBox[0][j]

                if m == -1:
                    continue

                xmin1 = predictions[0][m][1]
                ymin1 = predictions[0][m][0]
                xmax1 = predictions[0][m][3]
                ymax1 = predictions[0][m][2]

                iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1)

                if iou >= 0.45:
                    candidateBox[0][j] = -1

        detections = np.zeros((20, 6), np.float32)
        for i in range(0, vaildCnt):
            if candidateBox[0][i] == -1:
                continue
            if i >= 20:
                break

            n = candidateBox[0][i]

            xmin = max(0.0, min(1.0, predictions[0][n][1]))
            ymin = max(0.0, min(1.0, predictions[0][n][0]))
            xmax = max(0.0, min(1.0, predictions[0][n][3]))
            ymax = max(0.0, min(1.0, predictions[0][n][2]))

            detections[i] = [
                candidateBox[1][i] - 1,
                classScore[i],
                ymax,
                xmax,
                ymin,
                xmin
            ]

        return detections
