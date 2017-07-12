import tensorflow as tf
import numpy as np
import data, cPickle, csv, os
from tflearn.datasets import cifar10


ON = True
DATA_SIZE = 0
BATCH_SIZE = 128
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

def cifar_eval(sess):
	acc = 0.0
	for i in range(1, 11):
		image = x_test[0 + (i - 1) * 1000:1000 * i]
		label = y_test[0 + (i - 1) * 1000:1000 * i]
		acc = acc + accuracy.eval(session=sess, feed_dict={x: image, y: label, keep_prob: 1.0, training: False})
	return acc / 10

def next_batch(i):
        '''batch_start_idx = (i * BATCH_SIZE) % (DATA_SIZE - BATCH_SIZE)
        batch_end_idx = batch_start_idx + BATCH_SIZE
        xs = x_[batch_start_idx:batch_end_idx]
        ys = y_[batch_start_idx:batch_end_idx]'''
	get = np.random.choice(DATA_SIZE, 128, False)
	xs = x_[get]
	ys = y_[get]
        return xs, ys


if os.path.isfile('pre_data'):
        with open('pre_data', 'rb') as f:
                dic = cPickle.load(f)
                train, test = dic['train'], dic['test']
                x_, y_ = train['x'], train['y']
                x_test, y_test = test['x'], test['y']
                print('data loaded.')
else:
        mean = np.array([125.3, 123.0, 113.9])
        std = np.array([63.0, 62.1, 66.7])
        #load data
        data_dir = './cifar-10-batches-py/cifar-10-batches-py'
        meta = data.unpickle(data_dir + '/' + 'batches.meta')
        label_names = meta['label_names']
        label_count = len(label_names)
        train_files = ['data_batch_%d' % d for d in xrange(1, 6)]
        x, y = data.load_data(train_files, data_dir, label_count)
        x_test, y_test = data.load_data(['test_batch'], data_dir, label_count)
        x = x.astype("float")
        x_test = x_test.astype("float")
        #data pre-processing
        for i in xrange(len(x)):
                for j in xrange(32):
                        for k in xrange(32):
                                x[i][j][k] = (x[i][j][k] - mean) / std
        for i in xrange(len(x_test)):
                for j in xrange(32):
                        for k in xrange(32):
                                x_test[i][j][k] = (x_test[i][j][k] - mean) / std
        print('data pre-processing finished.')

DATA_SIZE = len(x_)

#define placeholder
x = tf.placeholder(tf.float32, [None, 32, 32, 3])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
training = tf.placeholder(tf.bool)
W_decay = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

#data augmentation
pad_image = tf.cond(training, lambda: tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, 40, 40), x), lambda: x)
crop_image = tf.cond(training, lambda: tf.map_fn(lambda img: tf.random_crop(img, [32, 32, 3]), pad_image), lambda: pad_image)
flip_image = tf.cond(training, lambda: tf.map_fn(lambda img: tf.image.random_flip_left_right(img), crop_image), lambda: crop_image)

#conv1
w_conv1 = weight([5, 5, 3, 192], std=0.01)
b_conv1 = tf.Variable(tf.random_normal([192], stddev=0.01, dtype=tf.float32))
relu_conv1 = tf.nn.relu(conv2d(flip_image, w_conv1) + b_conv1)
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
avgpool = tf.nn.avg_pool(relu_mlp32, ksize=[1, 8, 8, 1], strides=[1, 1, 1, 1], padding='VALID')
#flatten
flatten = tf.reshape(avgpool, [-1, 1 * 1 * 10])

#loss function & weight deacy
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=flatten))
tf.summary.scalar('loss', cross_entropy)
l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
#optimizer
train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(cross_entropy + l2 * W_decay)
#prediction
correct_prediction = tf.equal(tf.argmax(flatten, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('acc', accuracy)

#start session
sess = tf.InteractiveSession()
#init
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs/')

acct = []
loss = []

for i in xrange(1, 31281):
	batch = next_batch(i)
	_, lo, acc, summary = sess.run([train_step, cross_entropy, accuracy, merged], feed_dict={x: batch[0], y: batch[1], keep_prob: prob, training: ON, W_decay: 0.0001, learning_rate: 0.1})
	if i % 391 == 0:
		train_writer.add_summary(summary, i / 391)
		acct.append(cifar_eval(sess))
		loss.append(lo)
    		print("Epoch %3d, Train loss %.8f, Train acc %.8f, Test acc %.8f" % (i / 391, lo, acc, acct[i / 391 - 1]))

for i in xrange(31281, 47312):
        batch = next_batch(i)
        _, lo, acc, summary = sess.run([train_step, cross_entropy, accuracy, merged], feed_dict={x: batch[0], y: batch[1], keep_prob: prob, training: ON, W_decay: 0.0001, learning_rate: 0.01})
        if i % 391 == 0:
                train_writer.add_summary(summary, i / 391)
                acct.append(cifar_eval(sess))
		loss.append(lo)
                print("Epoch %3d, Train loss %.8f, Train acc %.8f, Test acc %.8f" % (i / 391, lo, acc, acct[i / 391 - 1]))

for i in xrange(47312, 64125):
        batch = next_batch(i)
        _, lo, acc, summary = sess.run([train_step, cross_entropy, accuracy, merged], feed_dict={x: batch[0], y: batch[1], keep_prob: prob, training: ON, W_decay: 0.0001, learning_rate: 0.001})
        if i % 391 == 0:
                train_writer.add_summary(summary, i / 391)
                acct.append(cifar_eval(sess))
		loss.append(lo)
                print("Epoch %3d, Train loss %.8f, Train acc %.8f, Test acc %.8f" % (i / 391, lo, acc, acct[i / 391 - 1]))

out = []
for i in acct:
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

print("Final test accuracy %g" % cifar_eval(sess))
print 'prob: ', prob
