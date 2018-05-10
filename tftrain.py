import tensorflow as tf
from readfile import *
import numpy as np
import time

def shuffle(X_train, y_labels):
    idx = np.random.permutation(X_train.shape[0])
    return X_train[idx], y_labels[idx]


start = time.time()
def tftrain(X_train, y_labels, X_test, y_test, K, n_epoch, batch_size):

    x_data = tf.placeholder(shape=[None, K], dtype=tf.float32)
    y_target = tf.placeholder(shape=[None, ], dtype=tf.int32)

    W = tf.Variable(tf.random_normal(shape=[K, 10]))
    b = tf.Variable(tf.random_normal(shape=[1, 10]))

    model_output = tf.add(tf.matmul(x_data, W), b)
    l2_norm = tf.norm(W)
    hinge = tf.reduce_sum(tf.maximum(0., 1. - tf.reduce_sum(tf.multiply(x_data, tf.gather(tf.transpose(W), y_target)), axis=1, keepdims=True) 
        - tf.gather(tf.transpose(b), y_target) + tf.matmul(x_data, W) + b ))

    loss = hinge + l2_norm
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

    n = X_train.shape[0]
    best_accuracy = 0
    W_best = 0
    b_best = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) 
        
        for i in range(n_epoch):
            x_batch, y_batch = shuffle(X_train, y_labels)
            batch_id = 0
            while (batch_id + batch_size) < n:
                _, loss_ = sess.run([optimizer, loss], feed_dict={x_data: x_batch[batch_id: (batch_id + batch_size)], y_target: y_batch[batch_id: (batch_id + batch_size)]})
                batch_id += batch_size
            print('Loss: ', float(loss_) / n)
            out_test = sess.run(model_output, feed_dict={x_data: X_test})
            y_pred = np.argmax(out_test, axis=1)
            count = 0.0
            for j in range(y_pred.shape[0]):
                if y_pred[j] != y_test[j]:
                    count +=1.0
            print('Fails: ', count)
            accuracy = 100.0 - count * 100.0 / y_pred.shape[0]
            print('Rate: ', i, ' :', accuracy)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                W_best, b_best = sess.run([W, b])

    return W_best, b_best, best_accuracy


y = readfile('./y_labels.txt', dtype=int)[0]
X = readfile('./X_train.txt', dtype=float)
msk = np.random.rand(X.shape[0]) < 0.8
X_train = X[msk]
y_labels = y[msk]
X_test = X[~msk]
y_test = y[~msk]

W_out, b_out, rate = tftrain(X_train, y_labels, X_test, y_test, K=150, n_epoch=1000, batch_size=512)

print('=========================')
print('Thoi gian train: ', time.time() - start)
print('Rate: ', rate)
f = open('./W.txt', 'w+')
for i in W_out:
    for j in i:
       f.write('%s ' %j)
    f.write('\n')

f = open('./b.txt', 'w+')
for i in b_out:
    for j in i:
       f.write('%s ' %j)
    f.write('\n')
