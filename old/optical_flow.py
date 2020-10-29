import random

import imutils
import numpy as np
import cv2
from imutils import paths

video_paths = sorted(list(paths.list_files("dataset\\optical\\")))
random.seed()
random.shuffle(video_paths)

for video_path in video_paths:
    cap = cv2.VideoCapture(video_path)

    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    while (1):
        ret, frame2 = cap.read()
        if not ret:
            break

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.9, 3, 15, 3, 5, 1.2, 0)

        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        rgb = imutils.resize(rgb, width=400)
        cv2.imshow('frame2', rgb)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('dataset\\frames\\opticalfb.png', frame2)
            cv2.imwrite('dataset\\frames\\opticalhsv.png', rgb)
        prvs = next

    cap.release()
    cv2.destroyAllWindows()
