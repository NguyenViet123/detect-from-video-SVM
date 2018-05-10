import os
import cv2
import numpy as np 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time
# return link to datas and labels respectively

start = time.time()
def readData(curDir):
    trainingset = curDir + 'trainingset'
    datas = []
    labels = []
    layer = 1
    for subfolder in os.listdir(trainingset):
        for fileData in os.listdir(trainingset + '/' + subfolder):
            datas.append(trainingset + '/' + subfolder + '/' + fileData)
            labels.append(layer)
        layer += 1

    return (datas, labels)   

linkDatas, y_labels = readData('./')

def allDesOfKeypoin(linkDatas):
    sift = cv2.xfeatures2d.SIFT_create()
    allDesOfKeypoin = []
    t = 0
    for linkData in linkDatas:
        key, des = sift.detectAndCompute(cv2.imread(linkData), None)
        if des is not None:
            for d in des:
                allDesOfKeypoin.append(d)
        else:
            y_labels[t] = -1
        t += 1

    return np.array(allDesOfKeypoin)

allDesOfKeypoin = allDesOfKeypoin(linkDatas)


def dictionary(allDesOfKeypoin, K):
    kmean = KMeans(n_clusters=K).fit(allDesOfKeypoin)
    return np.array(kmean.cluster_centers_)

K = 200
dictionary = dictionary(allDesOfKeypoin, K)

# featureEngin tung anh
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

def featureDataTraining(linkDatas):
    X_train = []
    for linkData in linkDatas:
        img = cv2.imread(linkData)
        X_train.append(featureEngin(img, dictionary))

    return np.array(X_train)

X_train = featureDataTraining(linkDatas)
#dictionary, X_train, y_labels

f = open('./dictionary.txt', 'w+')
for i in dictionary:
    for j in i:
       f.write('%s ' %j)
    f.write('\n')

f = open('./X_train.txt', 'w+')
for i in X_train:
    i = i.tolist()
    if i != -1:
        for j in i:
           f.write('%s ' %j)
        f.write('\n')

f = open('./y_labels.txt', 'w+')

for i in y_labels:
    if i != -1:
        print(i)
        f.write('%s ' %i)

print(time.time() - start)

#Time: 933.8556785583496