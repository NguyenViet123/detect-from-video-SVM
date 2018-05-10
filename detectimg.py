import cv2
from readfile import readfile
import numpy as np 
from scipy.spatial.distance import cdist

W = readfile('./W.txt', dtype=float)
b = readfile('./b.txt', dtype=float)
dictionary = readfile('./dictionary.txt', dtype=float)
K = 150
print(W.shape, b.shape)

def featureEngin(img, dictionary):
    sift = cv2.xfeatures2d.SIFT_create()
    key, des = sift.detectAndCompute(img, None)
    if des is not None:
        argmin = np.argmin(cdist(des, dictionary), axis=1)
        dataFe = np.zeros((K,), dtype=int)
        for i in argmin:
            dataFe[i] += 1
    else:
        return np.array(-1)

    return dataFe

def predict(img, W, b):
    data = featureEngin(img, dictionary)
    model = np.argmax(np.dot(data, W) + b)

    if model == 0:
        return 'bus'
    if model == 1:
        return 'car'
    if model == 2:
        return 'moto'
    if model == 3:
        return 'pedestrian'
    if model == 4:
        return 'sub_moto'
    if model == 5:
        return 'truck_head'
    if model == 6:
        return 'truc_tail'

# predict(cv2.imread('./t.png'), W, b)

# Video
import cv2
import numpy as np 

cap = cv2.VideoCapture('people-walking.mp4')

kernel_dil = np.ones((20,20), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.createBackgroundSubtractorMOG2()
font = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    # frame = frame[100:fshape[0]-100,:fshape[1]-100,:]

    if ret == True:
        fgmask = fgbg.apply(frame)
        # smoot Morphological Transformations
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
        fgmask = cv2.erode(fgmask, kernel, iterations=1)
        fgmask = cv2.dilate(fgmask, kernel_dil, iterations=1)
        # 

        # arg1: image (binary image)
        # CHAIN_APPROX_SIMPLE: lấy các điểm giói hạn
        # CHAIN_APPROX_NONE: lấy tất cả các điểm
        _, contours, hierarchy= cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # draw contours
        # cv2.drawContours(frame, contours, -1, (0,255,0), 3)

          #enumerate(): pic la chi so cua contour  
        
        for pic, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 3000:
                x,y,w,h = cv2.boundingRect(contour)
                img = cv2.rectangle(frame, (x,y), (x+w, y+h), (16, 78, 139),2)

                # hình chữ nhật xoay
                # rect = cv2.minAreaRect(contour)
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                # cv2.drawContours(frame, [box], 0, (0,0,255), 2)

                # Crop image
                imCrop = frame[y:y+h, x:x+w]
                pred_label = predict(imCrop, W, b)
                cv2.putText(img, pred_label,(x, y), font, 1,(255,255,255),2,cv2.LINE_AA)

                # cv2.imwrite('test' + str(pic) + '.jpg', imCrop)

                # roi_vehchile = fgmask[y:y-10+h+5, x:x-8+w+10]

        cv2.imshow('original', frame)

        if cv2.waitKey(30) &0xff ==27:
            break

    else:
        break


# img = cv2.imread('test.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# for pic, contour in enumerate(contours,1):
#     print(pic, '   ',contour)
# # print(contuors[0])
