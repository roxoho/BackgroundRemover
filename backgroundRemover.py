import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import os

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

segmentor = SelfiSegmentation()
#imgBg = cv2.imread("images/3.jfif")
#cv2.resize(imgBg,(480, 640))

listImg = os.listdir("images")
imgList = []
for imgPath in listImg:
    img = cv2.imread(f'images/{imgPath}')
    imgList.append(img)

indexImg = 0


while True:
    _, img = cap.read()
    img_out = segmentor.removeBG(img, imgList[indexImg], cutThreshold=0.8)

    imgStacked = cvzone.stackImages([img, img_out], 2, 1)

    cv2.imshow('webcam', imgStacked)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('a'):
        if indexImg>0:
            indexImg -= 1
    elif key == ord('d'):
        if indexImg < len(imgList) - 1:
            indexImg += 1

cap.release()
cv2.destroyAllWindows()
