import numpy as np

def readfile(filename, dtype):
    f = open(filename, 'r')
    arr = []
    while True:
        a = f.readline()
        if a == '':
            break
        a = a.split(' ')
        a =list(map(dtype, a[0:-1]))
        arr.append(a)
    
    return np.array(arr)
    


# 2479 diem du lieu
# y_labels = readfile('./y_labels.txt', dtype=int)[0]
# X_train = readfile('./X_train.txt', dtype=int)
# dictionary = readfile('./dictionary.txt', dtype=float)
