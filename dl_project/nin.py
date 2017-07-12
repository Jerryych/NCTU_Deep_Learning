import tensorflow as tf
import numpy as np
import data, cPickle, csv, os
from sklearn.model_selection import train_test_split


ON = True
DATA_SIZE = 0
BATCH_SIZE = 8
prob = 1.0

def weight(shape, std=0.05):
	init = tf.random_normal(shape, stddev=std, dtype=tf.float32)
	return tf.Variable(init)

def bias(shape):
	init = tf.constant(0, shape=shape, dtype=tf.float32)
	return tf.Variable(init)

def conv2d(x, w):
	return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

def maxpool3x3(x):
	return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

def eval(sess):
	acc = 0.0
	for i in range(1, 11):
		image = x_test[0 + (i - 1) * 10:10 * i]
		label = y_test[0 + (i - 1) * 10:10 * i]
		acc = acc + accuracy.eval(session=sess, feed_dict={x: image, y: label, keep_prob: 1.0, training: False})
	return acc / 10

def next_batch():
	get = np.random.choice(DATA_SIZE, BATCH_SIZE, False)
	xs = x_[get]
	ys = y_[get]
        return xs, ys


if os.path.isfile('pre_data'):
        with open('pre_data', 'rb') as f:
                dic = cPickle.load(f)
                x_, x_test, y_, y_test = train_test_split(dic['x'], dic['y'], test_size=0.1, random_state=42, stratify=dic['y'])
		print 'train:', len(x_)
		print 'test:', len(x_test)
                print('data loaded.')

DATA_SIZE = len(x_)

#define placeholder
x = tf.placeholder(tf.float32, [None, 96, 1366, 1])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)
W_decay = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

#conv1
w_conv1 = weight([5, 5, 1, 192], std=0.01)
b_conv1 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
relu_conv1 = tf.nn.relu(conv2d(x, w_conv1) + b_conv1)
#mlp1-1
w_mlp11 = weight([1, 1, 192, 160])
b_mlp11 = bias([160])
relu_mlp11 = tf.nn.relu(conv2d(relu_conv1, w_mlp11) + b_mlp11)
#mlp1-2
w_mlp12 = weight([1, 1, 160, 96])
b_mlp12 = bias([96])
relu_mlp12 = tf.nn.relu(conv2d(relu_mlp11, w_mlp12) + b_mlp12)
#maxpool1
maxpool1 = maxpool3x3(relu_mlp12)
#dropout1
dropout1 = tf.nn.dropout(maxpool1, keep_prob)

#conv2
w_conv2 = weight([5, 5, 96, 192])
b_conv2 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
relu_conv2 = tf.nn.relu(conv2d(dropout1, w_conv2) + b_conv2)
#mlp2-1
w_mlp21 = weight([1, 1, 192, 192])
b_mlp21 = bias([192])
relu_mlp21 = tf.nn.relu(conv2d(relu_conv2, w_mlp21) + b_mlp21)
#mlp2-2
w_mlp22 = weight([1, 1, 192, 192])
b_mlp22 = bias([192])
relu_mlp22 = tf.nn.relu(conv2d(relu_mlp21, w_mlp22) + b_mlp22)
#maxpool2
maxpool2 = maxpool3x3(relu_mlp22)
#dropout2
dropout2 = tf.nn.dropout(maxpool2, keep_prob)

#conv3
w_conv3 = weight([3, 3, 192, 192])
b_conv3 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
relu_conv3 = tf.nn.relu(conv2d(dropout2, w_conv3) + b_conv3)
#mlp3-1
w_mlp31 = weight([1, 1, 192, 192])
b_mlp31 = bias([192])
relu_mlp31 = tf.nn.relu(conv2d(relu_conv3, w_mlp31) + b_mlp31)
#mlp3-2
w_mlp32 = weight([1, 1, 192, 10])
b_mlp32 = bias([10])
relu_mlp32 = tf.nn.relu(conv2d(relu_mlp31, w_mlp32) + b_mlp32)
#avgpool
avgpool = tf.nn.avg_pool(relu_mlp32, ksize=[1, relu_mlp32.get_shape()[1], relu_mlp32.get_shape()[2], 1], strides=[1, 1, 1, 1], padding='VALID')
#flatten
flatten = tf.reshape(avgpool, [-1, 1 * 1 * 10])

#loss function & weight deacy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=flatten))
tf.summary.scalar('loss', cross_entropy)
l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
#optimizer
#train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(cross_entropy + l2 * W_decay)
train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy + l2 * W_decay)
#prediction
correct_prediction = tf.equal(tf.argmax(flatten, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#start session
sess = tf.InteractiveSession()
#init
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

for i in xrange(1, 9001):
	batch = next_batch()
	_, lo, acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch[0], y: batch[1], keep_prob: prob, training: ON, W_decay: 0.0001, learning_rate: 0.0001})
	if i % 113 == 0:
    		print("Epoch %3d, Train loss %.8f, Train acc %.8f, Test acc %.8f" % (i / 113, lo, acc, eval(sess)))

for i in xrange(9001, 13613):
        batch = next_batch()
        _, lo, acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch[0], y: batch[1], keep_prob: prob, training: ON, W_decay: 0.0001, learning_rate: 0.00001})
        if i % 113 == 0:
                print("Epoch %3d, Train loss %.8f, Train acc %.8f, Test acc %.8f" % (i / 113, lo, acc, eval(sess)))

for i in xrange(13613, 18451):
        batch = next_batch()
        _, lo, acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch[0], y: batch[1], keep_prob: prob, training: ON, W_decay: 0.0001, learning_rate: 0.0000001})
        if i % 113 == 0:
                print("Epoch %3d, Train loss %.8f, Train acc %.8f, Test acc %.8f" % (i / 113, lo, acc, eval(sess)))


print("Final test accuracy %g" % eval(sess))
print 'prob: ', prob

if not os.path.exists('model_nin'):
	os.makedirs('model_nin')
save_path = saver.save(sess, "model_nin/model.ckpt")
print("Model saved in file: %s" % save_path)
