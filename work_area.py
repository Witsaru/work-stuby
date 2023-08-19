import time
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

pTime = 0

while True:

    _, img = cap.read()
    hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    work_lower = np.array([136, 87, 111], np.uint8)
    work_upper = np.array([180, 255, 255], np.uint8)
    work_mask = cv2.inRange(hsvFrame, work_lower, work_upper)

    kernel = np.ones((5, 5), "uint8")

    work_mask = cv2.dilate(work_mask, kernel)
    res_work = cv2.bitwise_and(img, img,
                              mask=work_mask)

    contours, hierarchy = cv2.findContours(work_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 300):
            x, y, w, h = cv2.boundingRect(contour)
            imageFrame = cv2.rectangle(img, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 255), 2)

            cv2.putText(imageFrame, "Red Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255))

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    cv2.imshow("Image", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()