import cv2
import mediapipe as mp
import time


class poseDetector():
    def __init__(self, mode=False, bodyComplexity=0, smooth=True,
                 detectConfidence=0.5, trackConfidence=0.5):
        self.mode = mode
        self.bodyComplexity = bodyComplexity
        self.smooth = smooth
        self.detectionConfidence = detectConfidence
        self.trackConfidence = trackConfidence

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.bodyComplexity, self.smooth,
                                     self.detectionConfidence, self.trackConfidence)

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        # print(self.results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        lmList = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                height, width, channel = img.shape
                # print(id, lm)
                # get the actual pixel value = x_landmark_ratio * (width)
                # get the actual pixel value = x_landmark_ratio * (height)
                cx, cy = int(lm.x * width), int(lm.y * height)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList


def main():
    cap = cv2.VideoCapture('videos/video4.mp4')
    previous_time = 0
    detector = poseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.findPosition(img)
        if len(lmList[14]) != 0:
            print(lmList[14])
            cv2.circle(img, (lmList[14][1], lmList[14][2]), 15, (255, 255, 0), cv2.FILLED)

        current_time = time.time()
        fps = 1 / (current_time - previous_time)
        previous_time = current_time

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 0), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()