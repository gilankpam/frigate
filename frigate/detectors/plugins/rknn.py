import numpy as np
import cv2

import logging
from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig
from typing import Literal
from pydantic import Field

logger = logging.getLogger(__name__)

DETECTOR_KEY = "rknn"

def postprocess(output, confidence_thres=0.5, iou_thres=0.5):
    outputs = np.transpose(np.squeeze(output[0]))
    
    # Get the number of rows in the outputs array
    rows = outputs.shape[0]

    # Lists to store the bounding boxes, scores, and class IDs of the detections
    boxes = []
    scores = []
    class_ids = []

    # Calculate the scaling factors for the bounding box coordinates
    x_factor = 1
    y_factor = 1

    # Iterate over each row in the outputs array
    for i in range(rows):
        # Extract the class scores from the current row
        classes_scores = outputs[i][4:]
    
        # Find the maximum score among the class scores
        max_score = np.amax(classes_scores)

        # If the maximum score is above the confidence threshold
        if max_score >= confidence_thres:
            # Get the class ID with the highest score
            class_id = np.argmax(classes_scores)

            # Extract the bounding box coordinates from the current row
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            # skip box with zero area
            if w == 0 or h == 0:
                continue

            # Calculate the scaled coordinates of the bounding box
            x1 = int((x - w / 2) * x_factor)
            y1 = int((y - h / 2) * y_factor)
            x2 = x1 + int(w * x_factor)
            y2 = y1 + int(h * y_factor)

            # Add the class ID, score, and box coordinates to the respective lists
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x1, y1, x2, y2])

    # Apply non-maximum suppression to filter out overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)

    detections = []

    # Iterate over the selected indices after non-maximum suppression
    for i in indices:
        detections.append([
            boxes[i],
            scores[i],
            class_ids[i]
        ])

    # Return the modified input image
    return detections

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

        self.config = config

    def detect_raw(self, tensor_input):
        outputs = self.rknn_lite.inference(inputs=[tensor_input])
        
        det = postprocess(outputs[0])

        detections = np.zeros((20, 6), np.float32)
        for i, d in enumerate(det):
            if i >= 20:
                break

            x1, y1, x2, y2 = d[0][0], d[0][1], d[0][2], d[0][3]

            detections[i] = [
                d[2],
                float(1),
                y1 / self.config.model.height,  # y_min
                x1 / self.config.model.width,  # x_min
                y2 / self.config.model.height,  # y_max
                x2 / self.config.model.width,  # x_max
            ]

        return detections
