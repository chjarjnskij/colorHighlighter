from utils import *
import cv2

# STEP 1
path = "test.png"
img = cv2.imread(path)
cv2.imshow("Original image", img)

# Hsv for other colors
hsv_yellow = [20, 40, 185, 205, 255, 255]
hsv_petrol = [80, 100, 101, 121, 98, 178]
hsv_pink = [136, 156, 255, 255, 170, 250]

hsv_green = [37, 57, 186, 206, 255, 255]
hsv_red = [0, 0, 255, 255, 161, 241]
# Creating the contours for different colours
imgResult_yellow = detectColor(img, hsv_yellow, "yellow")
imgResult_petrol = detectColor(img, hsv_petrol, "petrol")
imgResult_pink = detectColor(img, hsv_pink, "pink")
imgResult_green = detectColor(img, hsv_green, "green")
imgResult_red = detectColor(img, hsv_red, "red")
# Displaying the contours
# what would be the best thing? do everything separate and then print them together and then associate them
# to a different entity
imgContours_yellow, contours_yellow = getContours(imgResult_yellow, img, showCanny=True, cThr=[100, 150], draw=True)
imgContours_petrol, contours_petrol = getContours(imgResult_petrol, img, showCanny=True, cThr=[100, 150], draw=True)
imgContours_pink, contours_pink = getContours(imgResult_pink, img, showCanny=True, cThr=[100, 150], draw=True)
imgContours_green, contours_green = getContours(imgResult_green, img, showCanny=True, cThr=[100, 150], draw=True)
imgContours_red, contours_red = getContours(imgResult_red, img, showCanny=True, cThr=[100, 150], draw=True)

cv2.imshow("imgContours_yellow", imgContours_yellow)
cv2.imshow("imgContours_petrol", imgContours_petrol)
cv2.imshow("imgContours_pink", imgContours_pink)
cv2.imshow("imgContours_green", imgContours_green)
cv2.imshow("imgContours_red", imgContours_red)

roiList_yellow = getRoi(img, contours_yellow)
roiDisplay(roiList_yellow, "yellow")
roiList_petrol = getRoi(img, contours_petrol)
roiDisplay(roiList_petrol, "petrol")
roiList_pink = getRoi(img, contours_pink)
roiDisplay(roiList_pink, "pink")
roiList_green = getRoi(img, contours_green)
roiDisplay(roiList_green, "green")
roiList_red = getRoi(img, contours_red)
roiDisplay(roiList_red, "red")

# STEP 2
# hsv = [37, 57, 186, 206, 255, 255]
# imgResult = detectColor(img, hsv)

# STEP 3 & 4
# imgContours, contours = getContours(imgResult, img, showCanny=True, cThr=[100, 150], draw=True)
# cv2.imshow("imgContours", imgContours)
# print(len(contours))

# STEP 5
# roiList = getRoi(img, contours)
# roiDisplay(roiList)

cv2.waitKey(0)  # delay 0
cv2.destroyAllWindows()

# STEP 6