import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture('videos/video2.mp4')
previous_time = 0

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            height, width, channel = img.shape
            print(id, lm)
            # get the actual pixel value = x_landmark_ratio * (width)
            # get the actual pixel value = x_landmark_ratio * (height)
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

    current_time = time.time()
    fps = 1/(current_time - previous_time)
    previous_time = current_time

    cv2.putText(img, str(int(fps)), (70,50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Image", img)
    cv2.waitKey(1)