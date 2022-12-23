import cv2
import numpy as np
import PoseModule
import pyautogui
import win32gui
import win32con

cap = cv2.VideoCapture(0)
detector = PoseModule.PoseDetector()
# 1 = in , 0 = out
dir = 1
count = 0
num = 1

while True:
    success, img = cap.read()
    # img = cv2.resize(img, (750, 600))
    img = detector.findPose(img, draw=False)
    lmlist = detector.findPoints(img)

    angle_1 = detector.findangle(11, 13, 15)

    per = np.interp(angle_1, (90, 170), (0, 100))
    # print(int(per))

    color_1 = (0, 0, 255)

    # judge std of  arm
    if per == 100:
        color_1 = (0, 255, 0)
        if dir == 0:
            count += 0.5
            dir = 1

    if per == 0:
        color_1 = (0, 255, 0)
        if dir == 1:
            count += 0.5
            dir = 0

    if count == num:
        pyautogui.press('space')
        num += 1

    cv2.rectangle(img, (0, 0), (100, 100), (255, 255, 255), cv2.FILLED)
    cv2.putText(img, str(int(count)), (12, 70),
               cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    detector.drawangle(img, 11, 13, 15, color_1)

    cv2.imshow("video", img)
    hwnd = win32gui.FindWindow(None, u"video")
    # 通过句柄窗口置顶
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                          win32con.SWP_NOMOVE | win32con.SWP_NOACTIVATE |
                          win32con.SWP_NOOWNERZORDER | win32con.SWP_SHOWWINDOW | win32con.SWP_NOSIZE)
    cv2.waitKey(1)


