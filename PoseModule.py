import cv2
import mediapipe as mp
import time
import math
import numpy as np


class poseDetector():

    # static_image_mode = False,
    # model_complexity = 1,
    # smooth_landmarks = True,
    # enable_segmentation = False,
    # smooth_segmentation = True,
    # min_detection_confidence = 0.5,
    # min_tracking_confidence = 0.5)

    def __init__(self, mode=False, model = 1, smooth=True,
                 enable_seg = False, smosegment = True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.model = model
        self.smooth = smooth
        self.enable_seg = enable_seg
        self.smosegment = smosegment
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model, self.smooth,
                                     self.enable_seg, self.smosegment,
                                     self.detectionCon, self.trackCon)
        # self.pose = self.mpPose.Pose()

    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True):
        self.lmList = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return self.lmList

    def findAngle(self, img, p1, p2, p3, draw=True):

        # Get the landmarks
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        x3, y3 = self.lmList[p3][1:]

        # Calculate the Angle
        angle = math.degrees(math.atan2(y3 - y2, x3 - x2) -
                             math.atan2(y1 - y2, x1 - x2))
        if angle < 0:
            angle += 360

        # print(angle)

        # Draw
        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (51, 255, 51), cv2.FILLED)
            #cv2.circle(img, (x1, y1), 15, (153, 0, 0), 2)
            cv2.circle(img, (x2, y2), 10, (51, 255, 51), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (153, 0, 0), 2)
            cv2.circle(img, (x3, y3), 10, (51, 255, 51), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (153, 0, 0), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (51, 0, 0), 2)
        return angle

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    scale_percent = 120  # percent of original size
    stepL = 0
    stepR = 0
    start_moveL = 0
    end_moveL = 0
    start_moveR = 0
    end_moveR = 0

    while True:
        success, img = cap.read()
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # print('Resized Dimensions : ', resized.shape)

        #cv2.imshow("Resized image", resized)
        img = detector.findPose(resized, draw=False)
        lmList = detector.findPosition(resized, draw=False)
        ##print(shoulder)
        if len(lmList) != 0:

            shoulder = math.dist(lmList[11][1:], lmList[12][1:])

            assembly = math.dist(lmList[15][1:], lmList[16][1:])
            print(assembly)

            p1 = (lmList[11][1] - 15, lmList[11][2] + 15)
            p2 = (lmList[11][1] + 15, lmList[11][2] + 15)
            p3 = (lmList[11][1], lmList[11][2] - 15)
            q1 = (lmList[12][1] - 15, lmList[12][2] + 15)
            q2 = (lmList[12][1] + 15, lmList[12][2] + 15)
            q3 = (lmList[12][1], lmList[12][2] - 15)

            if assembly > 180:
                angL = detector.findAngle(resized, 11, 13, 15)
                if angL > 210:
                    cv2.circle(resized, (lmList[11][1:]), 15, (0, 255, 0), 4)
                    if stepL == 1:
                        end_moveL = time.time()
                        stepL = 0
                        #print(f"total time_L: {end_moveL - start_moveL}")

                else:
                    cv2.line(resized, p1, p2, (0, 0, 255), 4)
                    cv2.line(resized, p2, p3, (0, 0, 255), 4)
                    cv2.line(resized, p1, p3, (0, 0, 255), 4)
                    if stepL == 0:
                        start_moveL = time.time()
                        stepL = 1

                angR = detector.findAngle(resized, 12, 14, 16)
                if angR < 165:
                    cv2.circle(resized, (lmList[12][1:]), 15, (0, 255, 0), 4)
                    if stepR == 1:
                        end_moveR = time.time()
                        stepR = 0
                        #print(f"total time_R: {end_moveR - start_moveR}")

                else:
                    cv2.line(resized, q1, q2, (0, 0, 255), 4)
                    cv2.line(resized, q2, q3, (0, 0, 255), 4)
                    cv2.line(resized, q1, q3, (0, 0, 255), 4)
                    if stepR == 0:
                        start_moveR = time.time()
                        stepR = 1

            else:
                detector.findAngle(resized, 11, 13, 15)
                detector.findAngle(resized, 12, 14, 16)
                cv2.rectangle(resized, (lmList[11][1] - 15, lmList[11][2] + 15), (lmList[11][1] + 15, lmList[11][2] - 15), (255, 0, 0), 4)
                cv2.rectangle(resized, (lmList[12][1] - 15, lmList[12][2] + 15), (lmList[12][1] + 15, lmList[12][2] - 15), (255, 0, 0), 4)

            #print(f"time move: {start_moveL} time stop: {end_moveL}")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime


        cv2.putText(resized, str(int(fps)), (70, 50), cv2.FONT_ITALIC, 1,
                    (0, 255, 0), 3)


        cv2.imshow("Image", resized)
        #cv2.imshow("ImageRGB", imgRGB)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()