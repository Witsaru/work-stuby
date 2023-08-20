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
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle

def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = poseDetector()
    scale_percent = 120  # percent of original size
    axisXleft = []
    axisYleft = []
    axisXright = []
    axisYright = []
    detaXleft = 0
    detaYleft = 0
    detaXright = 0
    detaYright = 0
    p1 = (100, 500)
    p2 = (150, 500)
    p3 = (125, 450)
    q1 = (600, 500)
    q2 = (650, 500)
    q3 = (625, 450)
    num = 0

    while True:
        success, img = cap.read()
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # print('Resized Dimensions : ', resized.shape)

        #cv2.imshow("Resized image", resized)
        img = detector.findPose(resized)
        lmList = detector.findPosition(resized, draw=False)
        if len(lmList) != 0:
            #print(f"left wrist (x, y): {lmList[15][1:]} right wrist (x, y): {lmList[16][1:]}")
            Asst = math.dist(lmList[15][1:], lmList[16][1:])
            if Asst > 132:
                axisXleft.append(lmList[15][2])
                print(axisXleft)
                for axisX in axisYleft:
                    detaXleft =  axisXleft[1] - axisXleft[0]
                    axisXleft.pop(0)
                    if detaXleft > 2 or detaXleft < -2:
                        cv2.line(resized, p1, p2, (0, 0, 255), 2)
                        cv2.line(resized, p2, p3, (0, 0, 255), 2)
                        cv2.line(resized, p1, p3, (0, 0, 255), 2)


                    else:
                        cv2.circle(resized, (125,500), 15, (0, 255, 0), 2)

                axisXright.append(lmList[16][1])
                if len(axisXright) > 1:
                    detaXright = axisXright[1] - axisXright[0]
                    axisXright.pop(0)

                    if detaXright > 4 or detaXright < -4:
                        cv2.line(resized, q1, q2, (0, 0, 255), 2)
                        cv2.line(resized, q2, q3, (0, 0, 255), 2)
                        cv2.line(resized, q1, q3, (0, 0, 255), 2)

                    elif -4 <= detaXright <= 4 :
                        cv2.circle(resized, (625,500), 15, (0, 255, 0), 2)
                #print(f"deta x: {detaX} Type:{type(detaX)} deta y: {detaY} ")
                #cv2.circle(img, (lmList[16][1], lmList[16][2]), 10, (0, 255, 0), cv2.FILLED)
                #cv2.circle(img, (lmList[15[1], lmList[15][2]), 10, (0, 255, 0), 2)


            elif Asst <= 132:
                cv2.rectangle(resized, (100,500), (150,450), (255, 0, 0), 2)
                cv2.rectangle(resized, (600, 500), (650, 450), (255, 0, 0), 2)

                #print(Asst)

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