import cv2
# import time
import mediapipe as mp
import math
import numpy as np
import os
import csv

x1 = 0
x2 = 0
x3 = 0
y1 = 0
y2 = 0
y3 = 0


class PoseDetector():
    def __init__(self, mode=False, complexity=1, smooth_lm=True,
                 enable=False, smooth=True, detection=0.5, track=0.5):
        self.mode = mode
        self.complexity = complexity
        self.smooth_lm = smooth_lm
        self.enable = enable
        self.smooth = smooth
        self.detection = detection
        self.track = track
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.complexity, self.smooth_lm,
                                     self.enable, self.smooth,
                                     self.detection, self.track)
        self.mpDraw = mp.solutions.drawing_utils
        # img = cv2.imread('photo/2.png',1)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(self.results.pose_landmarks)

        if draw:
            if self.results.pose_landmarks:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)

        return img

    def findPoints(self, img):
        self.lmlist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                x, y, z = int(w * lm.x), int(h * lm.y), int(w * lm.z)
                self.lmlist.append([id, x, y, z])

        return self.lmlist

    def findPoints_3D(self, img):
        self.lmlist_3D = []
        if self.results.pose_world_landmarks:
            for id, lm in enumerate(self.results.pose_world_landmarks.landmark):
                h, w, c = img.shape
                x, y, z = int(w * lm.x), int(h * lm.y), int(w * lm.z)
                self.lmlist_3D.append([id, x, y, z])

        return self.lmlist_3D

    def findangle(self, p1, p2, p3):
        global x1, x2, x3, y1, y2, y3
        if len(self.lmlist) != 0:
            x1 = self.lmlist[p1][1]
            x2 = self.lmlist[p2][1]
            x3 = self.lmlist[p3][1]

            y1 = self.lmlist[p1][2]
            y2 = self.lmlist[p2][2]
            y3 = self.lmlist[p3][2]

        angle = math.degrees(math.atan2(y3 - y2, x3 - x2)
                             - math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        return angle

    def drawangle(self, img, p1, p2, p3, color, draw=True):
        if draw:
            if len(self.lmlist) != 0:
                cv2.circle(img, (self.lmlist[p2][1], self.lmlist[p2][2]),
                           10, (255, 255, 255), cv2.FILLED)
                cv2.circle(img, (self.lmlist[p1][1], self.lmlist[p1][2]),
                           10, (255, 255, 255), cv2.FILLED)
                cv2.circle(img, (self.lmlist[p3][1], self.lmlist[p3][2]),
                           10, (255, 255, 255), cv2.FILLED)
                cv2.line(img, (self.lmlist[p2][1], self.lmlist[p2][2]),
                         (self.lmlist[p3][1], self.lmlist[p3][2]), color,
                         4)
                cv2.line(img, (self.lmlist[p2][1], self.lmlist[p2][2]),
                         (self.lmlist[p1][1], self.lmlist[p1][2]), color,
                         4)

    def findangle_z(self, p1, p2, p3):
        x = np.array((self.lmlist[p1][1] - self.lmlist[p2][1], self.lmlist[p1][2]
                      - self.lmlist[p2][2], self.lmlist[p1][3] - self.lmlist[p2][3]))
        y = np.array((self.lmlist[p3][1] - self.lmlist[p2][1], self.lmlist[p3][2]
                      - self.lmlist[p2][2], self.lmlist[p3][3] - self.lmlist[p2][3]))
        # 分别计算两个向量的模：
        l_x = np.sqrt(x.dot(x))
        l_y = np.sqrt(y.dot(y))
        # 计算两个向量的点积
        point = x.dot(y)
        # 计算夹角的cos值：
        cos_ = point / (l_x * l_y)
        # 求得夹角（弧度制）：
        angle_hu = np.arccos(cos_)
        # 转换为角度值：
        angle_d = angle_hu * 180 / np.pi

        return angle_d

        # if draw:
        #     if len(self.lmlist) != 0:
        #         cv2.circle(img, (x2, y2),
        #                 10, (255, 255, 255), cv2.FILLED)
        #         cv2.circle(img, (x1, y1),
        #                 10, (255, 255, 255), cv2.FILLED)
        #         cv2.circle(img, (x3, y3),
        #                 10, (255, 255, 255), cv2.FILLED)
        #         cv2.line(img, (x2, y2), (x3, y3), color, 4)
        #         cv2.line(img, (x2, y2), (x1, y1), color, 4)

    def findangle_3D(self, p1, p2, p3):
        x = np.array((self.lmlist_3D[p1][1] - self.lmlist_3D[p2][1], self.lmlist_3D[p1][2]
                      - self.lmlist_3D[p2][2], self.lmlist_3D[p1][3] - self.lmlist_3D[p2][3]))
        y = np.array((self.lmlist_3D[p3][1] - self.lmlist_3D[p2][1], self.lmlist_3D[p3][2]
                      - self.lmlist_3D[p2][2], self.lmlist_3D[p3][3] - self.lmlist_3D[p2][3]))
        # 分别计算两个向量的模：
        l_x = np.sqrt(x.dot(x))
        l_y = np.sqrt(y.dot(y))
        # 计算两个向量的点积
        point = x.dot(y)
        # 计算夹角的cos值：
        cos_ = point / (l_x * l_y)
        # 求得夹角（弧度制）：
        angle_hu = np.arccos(cos_)
        # 转换为角度值：
        angle_3D = angle_hu * 180 / np.pi

        return angle_3D

    # def drawangle_3D(self, img, p1, p2, p3, color, draw=True):
    #     if draw:
    #         if len(self.lmlist_3D) != 0:
    #             cv2.circle(img, (self.lmlist_3D[p2][1], self.lmlist_3D[p2][2]),
    #                       10, (255, 255, 255), cv2.FILLED)
    #             cv2.circle(img, (self.lmlist_3D[p1][1], self.lmlist_3D[p1][2]),
    #                       10, (255, 255, 255), cv2.FILLED)
    #             cv2.circle(img, (self.lmlist_3D[p3][1], self.lmlist_3D[p3][2]),
    #                       10, (255, 255, 255), cv2.FILLED)
    #             cv2.line(img, (self.lmlist_3D[p2][1], self.lmlist_3D[p2][2]), (self.lmlist_3D[p3][1], self.lmlist_3D[p3][2]), color,
    #                     4)
    #             cv2.line(img, (self.lmlist_3D[p2][1], self.lmlist_3D[p2][2]), (self.lmlist_3D[p1][1], self.lmlist_3D[p1][2]), color,
    #                     4)


# class cvtPoselm2csv():
#     def __init__(self):


#
# cv2.imshow('image',img)
# cv2.waitKey(0)


# if __name__ == "__main__":
#     cap = cv2.VideoCapture('video/10.mp4')
#     # cap = cv2.VideoCapture(0)
#     detector = PoseDetector()
#     while True:
#         fps = cap.get(cv2.CAP_PROP_FPS)
#         success, img = cap.read()
#         img = cv2.resize(img, (750, 600))
#         img = detector.findPose(img)
#         lmlist = detector.findPoints(img)
#         # print(lmlist)
#         if lmlist:
#             cv2.circle(img, (lmlist[14][1], lmlist[14][2]), 10, (250, 0, 0), cv2.FILLED)
#         # results = pose.process(img)
#         # img = cv2.cv2tColor(img,cv2.COLOR_BGR2RGB)
#         cv2.putText(img, str(int(fps)), (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255))
#
#         cv2.imshow("video", img)
#         cv2.waitKey(1)
