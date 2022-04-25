import cv2
import numpy as np

# function to mask and detect colours
def detectColor(img, hsv, color):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # cv2.imshow(hsv, "imgHSV")
    lower = np.array([hsv[0], hsv[2], hsv[4]])
    upper = np.array([hsv[1], hsv[3], hsv[5]])
    mask = cv2.inRange(imgHSV, lower, upper)
    cv2.imshow("mask_"+color, mask)
    imgResult = cv2.bitwise_and(img, img, mask=mask)
    cv2.imshow("imgResult_"+color, imgResult)
    return imgResult

# function which creates the contours around the masked stuff
def getContours(img, imgDraw, cThr=[100, 100], showCanny=False, draw=True):
    imgDraw = imgDraw.copy()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 1)
    imgCanny = cv2.Canny(imgBlur, cThr[0], cThr[1])
    kernel = np.array((10, 10))
    imgDial = cv2.dilate(imgCanny, kernel, iterations=1)
    imgClose = cv2.morphologyEx(imgDial, cv2.MORPH_CLOSE, kernel)

    # if showCanny: cv2.imshow("Canny", imgClose)
    contours, hiearchy = cv2.findContours(imgClose, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalCountours = []
    for i in contours:
        area = cv2.contourArea(i)
        peri = cv2.arcLength(i, True)
        approx = cv2.approxPolyDP(i, 0.02 * peri, True)
        bbox = cv2.boundingRect(approx)
        finalCountours.append([len(approx), area, approx, bbox, i])

    finalCountours = sorted(finalCountours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalCountours:
            x, y, w, h = con[3]
            cv2.rectangle(imgDraw, (x, y), (x + w, y + h), (255, 0, 255), 3)
            # cv2.drawContours(imgDraw, con[4], -1, (0,0,255), 2)
    return imgDraw, finalCountours

# creates a list of contours
def getRoi(img, contours):
    # roiList = []
    for con in contours:
        x, y, w, h = con[3]
        roiList = []
        for con in contours:
            x, y, w, h = con[3]
            roiList.append(img[y:y + h, x:x + w])
        return roiList

# display all the contours
def roiDisplay(roiList, color):
    for x, roi in enumerate(roiList):
        roi = cv2.resize(roi, (0, 0), None, 2, 2)
        cv2.imshow(str(x)+'_'+color, roi)  # reasons why prints for one color only