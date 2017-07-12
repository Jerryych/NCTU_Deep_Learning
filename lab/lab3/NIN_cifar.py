#for deep learning course in NCTU 2017
#if you have questions, mail to followwar@gmail.com
#the code demonstrate the NIN arch on MNIST
#It achieved 0.41% test error (the original paper reported 0.47% test error)

from __future__ import print_function
import tensorflow as tf
from tflearn.datasets import cifar10
import numpy as np
import csv

DATA_SIZE = 0
BATCH_SIZE = 128

(X, Y), (X_test, Y_test) = cifar10.load_data(one_hot=True)
DATA_SIZE = len(X)



# define functions
def weight_variable(shape):
    initial = tf.random_normal(shape, stddev=0.05, dtype=tf.float32)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0, shape=shape,dtype=tf.float32)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_3x3(x):
    return tf.nn.max_pool(x, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')


#define placeholder
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)

pad_image = tf.cond(training, lambda: tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, 40, 40), x), lambda: x)
crop_image = tf.cond(training, lambda: tf.map_fn(lambda img: tf.random_crop(img, [32, 32, 3]), pad_image), lambda: pad_image)
flip_image = tf.cond(training, lambda: tf.map_fn(lambda img: tf.image.random_flip_left_right(img), crop_image), lambda: crop_image)
## conv1 layer ##
W_conv1 = weight_variable([5, 5, 3, 192])
b_conv1 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
output = tf.nn.relu(conv2d(flip_image, W_conv1) + b_conv1)
## MLP-1-1##
W_MLP11 = weight_variable([1, 1, 192, 160])
b_MLP11 = bias_variable([160])
output = tf.nn.relu(conv2d(output, W_MLP11) + b_MLP11)
## MLP-1-2##
W_MLP12 = weight_variable([1, 1, 160, 96])
b_MLP12 = bias_variable([96])
output = tf.nn.relu(conv2d(output, W_MLP12) + b_MLP12)
##Max pooling##
output = max_pool_3x3(output)                     
## dropout ##
output = tf.nn.dropout(output, keep_prob)

## conv2 layer ##
W_conv2 = weight_variable([5, 5, 96, 192])
b_conv2 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
output = tf.nn.relu(conv2d(output, W_conv2) + b_conv2)
## MLP-2-1##
W_MLP21 = weight_variable([1, 1, 192, 192])
b_MLP21 = bias_variable([192])
output = tf.nn.relu(conv2d(output, W_MLP21) + b_MLP21)
## MLP-2-2##
W_MLP22 = weight_variable([1, 1, 192, 192])
b_MLP22 = bias_variable([192])
output = tf.nn.relu(conv2d(output, W_MLP22) + b_MLP22)
##Max pooling##
output = max_pool_3x3(output)    
## dropout ##
output = tf.nn.dropout(output, keep_prob)

## conv3 layer ##
W_conv3 = weight_variable([3, 3, 192, 192])
b_conv3 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
output = tf.nn.relu(conv2d(output, W_conv3) + b_conv3)
## MLP-2-1##
W_MLP31 = weight_variable([1, 1, 192, 192])
b_MLP31 = bias_variable([192])
output = tf.nn.relu(conv2d(output, W_MLP31) + b_MLP31)
## MLP-2-2##
W_MLP32 = weight_variable([1, 1, 192, 10])
b_MLP32 = bias_variable([10])
output = tf.nn.relu(conv2d(output, W_MLP32) + b_MLP32)
##global average##
output = tf.nn.avg_pool(output, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
# [n_samples, 1, 1, 10] ->> [n_samples, 1*1*10]
output = tf.reshape(output, [-1, 1 * 1 * 10])

#this code demostrate how to schedule step-wise learning rate in a simple way
learning_rate = tf.placeholder(tf.float32)
# the loss function 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
# optimizer SGD
train_step = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cross_entropy)
# prediction
correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# start session
sess = tf.InteractiveSession()
# initize variable 
sess.run(tf.global_variables_initializer())

loss = []
acc = []

### TESTING ###
def cifar_eval(sess):
  acc = 0.0
  for i in range(1,11):  
   image = X_test[0 + (i - 1) * 1000:1000 * i]
   label = Y_test[0 + (i - 1) * 1000:1000 * i]
   acc= acc + accuracy.eval(session=sess, feed_dict={x: image, y_: label, keep_prob: 1.0, training: False})
  return acc/10
##############

def next_batch(i):
	batch_start_idx = (i * BATCH_SIZE) % (DATA_SIZE - BATCH_SIZE)
        batch_end_idx = batch_start_idx + BATCH_SIZE
        xs = X[batch_start_idx:batch_end_idx]
	ys = Y[batch_start_idx:batch_end_idx]
	return xs, ys

on = True

for i in range(1, 31281): #1~80 epochs
  batch = next_batch(i)
  _, lo = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, learning_rate: 0.05, training: on})  #learning rate 0.1
  ## test every 100 step
  if i % 391 == 0:
    loss.append(lo)
    acc.append(cifar_eval(sess))
    print("epoch %d, Test accuracy %g, Loss %g"%(i / 391, acc[i / 391 - 1], lo))

for i in range(31281, 47312): #81~121 epochs
  batch = next_batch(i)
  _, lo = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, learning_rate: 0.01, training: on})  #learning rate 0.01
  ## test every 100 step
  if i % 391 == 0:
    loss.append(lo)
    acc.append(cifar_eval(sess))
    print("epoch %d, Test accuracy %g, Loss %g"%(i / 391,  acc[i / 391 - 1], lo))

for i in range(47312, 54741):#64125): #122~164 epochs
  batch = next_batch(i)
  _, lo = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, learning_rate: 0.002, training: on}) ##learning rate 0.001
  ## test every 100 step
  if i % 391 == 0:
    loss.append(lo)
    acc.append(cifar_eval(sess))
    print("epoch %d, Test accuracy %g, Loss %g"%(i / 391,  acc[i / 391 - 1], lo))

for i in range(54741, 64125): #122~164 epochs
  batch = next_batch(i)
  _, lo = sess.run([train_step, cross_entropy], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, learning_rate: 0.0004, training: on}) ##learning rate 0.001
  ## test every 100 step
  if i % 391 == 0:
    loss.append(lo)
    acc.append(cifar_eval(sess))
    print("epoch %d, Test accuracy %g, Loss %g"%(i / 391,  acc[i / 391 - 1], lo))

out = []
for i in acc:
	out.append([i])

with open('acc.csv', 'w') as file:
	w = csv.writer(file)
	w.writerows(out)

out = []
for i in loss:
	out.append([i])

with open('loss.csv', 'w') as file:
	w = csv.writer(file)
	w.writerows(out)

	
##final testing
print("final, Test accuracy %g"%(cifar_eval(sess)))
