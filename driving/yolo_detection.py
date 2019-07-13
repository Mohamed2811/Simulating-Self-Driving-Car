import os
import time

import cv2
import numpy as np
# load our saved model

# helper classes
from data_collection.img_process import img_process
from data_collection.key_cap import key_check
# YOLO algorithm
from object_detection.object_detect import yolo_detection
from training.utils import preprocess




def main():

    while True:
        screen, resized, speed, direct = img_process("Grand Theft Auto V")
        radar = cv2.cvtColor(resized[206:226, 25:45, :], cv2.COLOR_RGB2BGR)[:, :, 2:3]
        resized = preprocess(resized)
        left_line_color = [0, 255, 0]
        right_line_color = [0, 255, 0]

        yolo_screen, color_detected, obj_distance = yolo_detection(screen, direct)
        cv2.imshow("Driving-mode", yolo_screen)
        cv2.waitKey(1)

if __name__ == '__main__':
    main()
