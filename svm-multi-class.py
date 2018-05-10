from __future__ import division, print_function, unicode_literals
import numpy as np
from mnist import MNIST
import tensorflow as tf
import os
import time
import cv2
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def read_mnist(filname):
	mndata = MNIST(filename)
	(x_train, y_train) = mndata.load_training()
	(x_test, y_test) = mndata.load_testing()
	x_train = np.asarray(x_train)
	y_train = np.asarray(y_train)
	x_test = np.asarray(x_test)
	y_test = np.asarray(y_test)
	return (x_train, y_train, x_test, y_test)

def random_batch(x_train, y_train):
	idx = np.random.permutation(x_train.shape[0])
	x_bar = x_train[idx]
	y_bar = y_train[idx]
	return x_bar, y_bar

def svm_multi_class(x_train, y_train, x_test, y_test, n_epoch, batch_size):
	x_data = tf.placeholder(shape=[batch_size, 784], dtype=tf.float32)
	y_target = tf.placeholder(shape=[batch_size, ], dtype=tf.int32)

	W = tf.Variable(tf.random_normal(shape=[784, 10]))
	b = tf.Variable(tf.random_normal(shape=[1, 10]))
	model_output = tf.add(tf.matmul(x_data, W), b)
	l2_norm = tf.norm(W)
	hinge = tf.reduce_sum(tf.maximum(0., 1. - tf.reduce_sum(tf.multiply(x_data, tf.gather(tf.transpose(W), y_target)), axis = 1, keepdims = True)
	- tf.gather(tf.transpose(b), y_target) + tf.matmul(x_data, W) + b ))
	loss = hinge + l2_norm
	opt = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

	n = x_train.shape[0]
	best_accuracy = 0
	W_best = 0
	b_best = 0
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(n_epoch):
			x_batch, y_batch = random_batch(x_train, y_train)
			batch_id = 0
			while(batch_id + batch_size) < n:
				_, loss_ = sess.run([opt, loss], feed_dict = {x_data : x_batch[batch_id: (batch_id + batch_size)], y_target : y_batch[batch_id: (batch_id + batch_size)]})
				batch_id += batch_size
			print('Loss: ', float(loss_) / n)
			out_test = sess.run(model_output, feed_dict = {x_data : x_test})
			y_pred = np.argmax(out_test, axis = 1)
			count = 0.0
			for j in range(y_pred.shape[0]):
				if y_pred[j] != y_test[j]:
					count += 1.0
			print(count, '!==', y_pred.shape[0])
			accuracy = 100.0 - count * 100.0 / y_pred.shape[0]
			print('Ti le cua  ',i ,'la:', accuracy)
			if accuracy > best_accuracy:
				best_accuracy = accuracy
				W_best, b_best = sess.run([W, b])

	return (W_best, b_best)

def read_file(folder):
	list_filenames = os.listdir(folder)
	data = np.zeros((1, 784))
	for filename in list_filenames:
		img = cv2.imread((folder + '/'+filename))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		img = np.reshape(img, (1, 784))
		data = np.concatenate((data, img), axis = 0)
	return data[1:, :]

def predict(data, W, b):
	for i in range(data.shape[0]):
		label = np.argmax(np.dot(data[i], W) + b, axis = 1)
		img = data[i].reshape((28, 28))
		plt.title("So:" + str(label))
		plt.imshow(img)
		plt.show()

filename = './MNIST'
folder = 'number'

data = read_file(folder)

(x_train, y_train, x_test, y_test) = read_mnist(filename)	
(W_best, b_best) = svm_multi_class(x_train, y_train, x_test[0:512], y_test[0:512], 10, 512)

predict(data, W_best, b_best)






