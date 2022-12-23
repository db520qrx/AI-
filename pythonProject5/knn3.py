
import cv2

import PoseModule


cap = cv2.VideoCapture('video/11.mp4')
# cap = cv.VideoCapture(0)
detector = PoseModule.PoseDetector()

while True:
    # fps = cap.get(cv2.CAP_PROP_FPS)
    success, img = cap.read()
    # img = cv2.resize(img, (750,600))
    img = detector.findPose(img, draw=False)
    lmlist = detector.findPoints(img)
