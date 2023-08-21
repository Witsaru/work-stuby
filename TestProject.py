import cv2
import time
import PoseModule as pm
import math
import paho.mqtt.publish as publish


def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = pm.poseDetector()
    scale_percent = 120  # percent of original size
    stepL = 0
    stepR = 0
    stepA = 0
    start_moveL = 0
    end_moveL = 0
    start_moveR = 0
    end_moveR = 0
    start_moveA = 0
    end_moveA = 0

    while True:
        success, img = cap.read()
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # print('Resized Dimensions : ', resized.shape)

        # cv2.imshow("Resized image", resized)
        img = detector.findPose(img, draw=False)
        lmList = detector.findPosition(img, draw=False)
        ##print(shoulder)
        if len(lmList) != 0:

            shoulder = math.dist(lmList[11][1:], lmList[12][1:])

            assembly = math.dist(lmList[15][1:], lmList[16][1:])
            #print(assembly)

            p1 = (lmList[11][1] - 15, lmList[11][2] + 15)
            p2 = (lmList[11][1] + 15, lmList[11][2] + 15)
            p3 = (lmList[11][1], lmList[11][2] - 15)
            q1 = (lmList[12][1] - 15, lmList[12][2] + 15)
            q2 = (lmList[12][1] + 15, lmList[12][2] + 15)
            q3 = (lmList[12][1], lmList[12][2] - 15)

            if assembly > 170:
                if stepA == 1:
                    end_moveA = time.time()
                    stepA = 0
                    #print(f"total time_R: {end_moveA - start_moveA}")
                    publish.single("Work_stuby/Assembly_time", f"{end_moveA - start_moveA} sec",
                                   hostname="127.0.0.1")
                angL = detector.findAngle(img, 11, 13, 15)
                publish.single("Work_stuby/Left", 360 - int(angL), hostname="127.0.0.1")
                if angL > 220:
                    cv2.circle(img, (lmList[11][1:]), 15, (0, 255, 0), 4)
                    if stepL == 1:
                        end_moveL = time.time()
                        stepL = 0
                        #print(f"total time_L: {end_moveL - start_moveL}")
                        publish.single("Work_stuby/Time_search_left", f"{end_moveL - start_moveL} sec", hostname="127.0.0.1")

                else:
                    cv2.line(img, p1, p2, (0, 0, 255), 4)
                    cv2.line(img, p2, p3, (0, 0, 255), 4)
                    cv2.line(img, p1, p3, (0, 0, 255), 4)
                    if stepL == 0:
                        start_moveL = time.time()
                        stepL = 1

                angR = detector.findAngle(img, 12, 14, 16)
                publish.single("Work_stuby/Right", int(angR), hostname="127.0.0.1")
                if angR < 160:
                    cv2.circle(img, (lmList[12][1:]), 15, (0, 255, 0), 4)
                    if stepR == 1:
                        end_moveR = time.time()
                        stepR = 0
                        #print(f"total time_R: {end_moveR - start_moveR}")
                        publish.single("Work_stuby/Time_search_right", f"{end_moveR - start_moveR} sec",
                                       hostname="127.0.0.1")

                else:
                    cv2.line(img, q1, q2, (0, 0, 255), 4)
                    cv2.line(img, q2, q3, (0, 0, 255), 4)
                    cv2.line(img, q1, q3, (0, 0, 255), 4)
                    if stepR == 0:
                        start_moveR = time.time()
                        stepR = 1

            else:
                detector.findAngle(img, 11, 13, 15)
                detector.findAngle(img, 12, 14, 16)
                cv2.rectangle(img, (lmList[11][1] - 15, lmList[11][2] + 15),
                              (lmList[11][1] + 15, lmList[11][2] - 15), (255, 0, 0), 4)
                cv2.rectangle(img, (lmList[12][1] - 15, lmList[12][2] + 15),
                              (lmList[12][1] + 15, lmList[12][2] - 15), (255, 0, 0), 4)

                if stepA == 0:
                    start_moveA = time.time()
                    stepA = 1

            # print(f"time move: {start_moveL} time stop: {end_moveL}")

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_ITALIC, 1,
                    (0, 255, 0), 3)

        cv2.imshow("Image", img)
        # cv2.imshow("ImageRGB", imgRGB)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

main()
