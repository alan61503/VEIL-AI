import cv2
import numpy as np

def classify_plate_color(plate_img):
    hsv = cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV)

    # Define yellow color range
    yellow_lower = np.array([15, 80, 80])
    yellow_upper = np.array([35, 255, 255])

    mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    yellow_ratio = cv2.countNonZero(mask) / (plate_img.shape[0] * plate_img.shape[1])

    if yellow_ratio > 0.3:
        return "Taxi"
    else:
        return "Private"
