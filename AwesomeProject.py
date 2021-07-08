import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('videos/video3.mp4')
previous_time = 0
detector = pm.poseDetector()

while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPosition(img)
    if len(lmList[28]) != 0:
        print(lmList[1])
        cv2.circle(img, (lmList[28][1], lmList[28][2]), 15, (0, 0, 255), cv2.FILLED)

    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)