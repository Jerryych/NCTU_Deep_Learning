import tensorflow as tf
import numpy as np
import cPickle, csv
from vgg19_cifar10 import VGG19_cifar10


MODE = 0
BN = True
DATA_SIZE = 0
BATCH_SIZE = 128

print 'MODE: ', MODE
print 'BN: ', BN

def cifar_eval(sess):
        acc = 0.0
        for i in range(1, 11):
                image = x_test[0 + (i - 1) * 1000:1000 * i]
                label = y_test[0 + (i - 1) * 1000:1000 * i]
                acc = acc + accuracy.eval(session=sess, feed_dict={x: image, y: label, data_aug: False, keep_prob: 1.0, bn_control: False})
        return acc / 10

def next_batch(i):
        get = np.random.choice(DATA_SIZE, BATCH_SIZE, False)
        xs = x_[get]
        ys = y_[get]
        return xs, ys

if MODE == 0:
	with open('pre_data_bgr', 'rb') as f:
		dic = cPickle.load(f)
		train, test = dic['train'], dic['test']
		x_, y_ = train['x'], train['y']
		x_test, y_test = test['x'], test['y']
		print('bgr data loaded.')
else:
	with open('pre_data_rgb', 'rb') as f:
                dic = cPickle.load(f)
                train, test = dic['train'], dic['test']
                x_, y_ = train['x'], train['y']
                x_test, y_test = test['x'], test['y']
                print('rgb data loaded.')

DATA_SIZE = len(x_)

with tf.Session() as sess:
	#placeholder
	x = tf.placeholder(tf.float32, [None, 32, 32, 3])
	y = tf.placeholder(tf.float32, [None, 10])
	data_aug = tf.placeholder(tf.bool)
	keep_prob = tf.placeholder(tf.float32)
	bn_control = tf.placeholder(tf.bool)
	W_decay = tf.placeholder(tf.float32)
	learning_rate = tf.placeholder(tf.float32)

	#build model
	vgg = VGG19_cifar10(MODE, BN)
	vgg.build(x, data_aug, keep_prob, bn_control)
	
	#loss
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=vgg.fc8))
	l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
	#optimizer
	train_step = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(cross_entropy + l2 * W_decay)
	#prediction
	correct_prediction = tf.equal(tf.argmax(vgg.fc8, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	#init varialbe
	sess.run(tf.global_variables_initializer())

	acct = []
	loss = []
	
	for i in xrange(1, 31281):
		batch = next_batch(i)
		_, lo, acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch[0], y: batch[1], data_aug: True, keep_prob: 0.5, bn_control: True, W_decay: 0.0001, learning_rate: 0.01})
		if i % 391 == 0:
			acct.append(cifar_eval(sess))
			loss.append(lo)
			print("Epoch %3d, Train loss %.8f, Train acc %.8f, Test acc %.8f" % (i / 391, lo, acc, acct[i / 391 - 1]))

	for i in xrange(31281, 47312):
		batch = next_batch(i)
		_, lo, acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch[0], y: batch[1], data_aug: True, keep_prob: 0.5, bn_control: True, W_decay: 0.0001, learning_rate: 0.001})
		if i % 391 == 0:
			acct.append(cifar_eval(sess))
			loss.append(lo)
			print("Epoch %3d, Train loss %.8f, Train acc %.8f, Test acc %.8f" % (i / 391, lo, acc, acct[i / 391 - 1]))

	for i in xrange(47312, 64125):
		batch = next_batch(i)
		_, lo, acc = sess.run([train_step, cross_entropy, accuracy], feed_dict={x: batch[0], y: batch[1], data_aug: True, keep_prob: 0.5, bn_control: True, W_decay: 0.0001, learning_rate: 0.0001})
		if i % 391 == 0:
			acct.append(cifar_eval(sess))
			loss.append(lo)
			print("Epoch %3d, Train loss %.8f, Train acc %.8f, Test acc %.8f" % (i / 391, lo, acc, acct[i / 391 - 1]))

	filename = str(MODE) + ('_bn' if BN else '')

	out = []
	out.append(['acc_' + filename])
	for i in acct:
		out.append([i])

	with open('acc_' + filename + '.csv', 'w') as file:
		w = csv.writer(file)
		w.writerows(out)

	out = []
	out.append(['loss_' + filename])
	for i in loss:
		out.append([i])

	with open('loss_' + filename + '.csv', 'w') as file:
		w = csv.writer(file)
		w.writerows(out)

	print 'MODE: ', MODE
	print 'BN: ', BN
	print("Final test accuracy %g" % cifar_eval(sess))
