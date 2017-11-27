import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import imutils

original = cv2.imread('img/172cm-20171115-130715.jpg') # 샘플사진 입력
# image = imutils.rotate_bound(original, 270)
(h, w) = image.shape[:2]

net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
net.setInput(blob)
detections = net.forward()

found = []
for i in np.arange(0, detections.shape[2]):
    if 15 == detections[0,0,i,1]:
        confidence = detections[0,0,i,2]
        if confidence > 0.2:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            box = np.array([[startX, startY, endX, endY]])
            if len(found) == 0:
                found = box
            else:
                found = np.append(found, box, axis=0)

    print(found)
    height = 0
    lastEndY = 0
    for ( startX, startY, endX, endY) in found:
        if height < endY - startY:
            lastEndY = 2464 - endY
            height = endY - startY
    if lastEndY < 0:
        lastEndY = 0 
    y = 0.648496 * lastEndY
    
    # 바뀐 카메라 환경 정보
    cameraheight = 1300
    startpoint = 2300
    
    # 카메라 각도 90도 기준 
    dist = ((y * startpoint) / ( cameraheight - y)) + startpoint # 거리계산 이론식
    dist = dist + (y * 1.3129) - 21  # 값보정
    
    result = 0.000425 * dist * height # 키계산 이론식
    result = (result + (dist * 0.0078)) * 0.826 # 값보정
    
    print("dist:{}, lastEndY:{}, height:{}, result:{}".format(dist, lastEndY, height, result))

plt.subplot(111)
plt.imshow(image)
plt.title('image')
plt.xticks([])
plt.yticks([])
plt.show()

