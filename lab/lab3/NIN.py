import tflearn
import numpy as np
import os
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, flatten
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import batch_normalization
from tflearn.data_augmentation import ImageAugmentation
from tflearn.layers.estimator import regression
import tflearn.optimizers
import data, cPickle


if os.path.isfile('pre_data'):
	with open('pre_data', 'rb') as f:
		dic = cPickle.load(f)
		train, test = dic['train'], dic['test']
		x, y = train['x'], train['y']
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
#data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop((32, 32), 4)
#input
network = input_data(shape=[None, 32, 32, 3], data_augmentation=img_aug)
#conv1
network = conv_2d(network, 192, 5, activation='relu', weights_init=tflearn.initializations.normal(stddev=0.01), weight_decay=0.0001)
network = conv_2d(network, 160, 1, activation='relu', weights_init=tflearn.initializations.normal(stddev=0.05), weight_decay=0.0001)
network = conv_2d(network, 96, 1, activation='relu', weights_init=tflearn.initializations.normal(stddev=0.05), weight_decay=0.0001)
network = max_pool_2d(network, 3, strides=2)
network = dropout(network, 0.5)
#conv2
network = conv_2d(network, 192, 5, activation='relu', weights_init=tflearn.initializations.normal(stddev=0.05), weight_decay=0.0001)
network = conv_2d(network, 192, 1, activation='relu', weights_init=tflearn.initializations.normal(stddev=0.05), weight_decay=0.0001)
network = conv_2d(network, 192, 1, activation='relu', weights_init=tflearn.initializations.normal(stddev=0.05), weight_decay=0.0001)
network = max_pool_2d(network, 3, strides=2)
network = dropout(network, 0.5)
#conv3
network = conv_2d(network, 192, 3, activation='relu', weights_init=tflearn.initializations.normal(stddev=0.05))
network = conv_2d(network, 192, 1, activation='relu', weights_init=tflearn.initializations.normal(stddev=0.05))
network = conv_2d(network, 10, 1, activation='relu', weights_init=tflearn.initializations.normal(stddev=0.05))
network = avg_pool_2d(network, 8, padding='valid')
network = flatten(network)
network = regression(network, optimizer=tflearn.optimizers.Momentum(learning_rate=0.1, lr_decay=0.1, decay_step=32000, staircase=True), loss='softmax_categorical_crossentropy',learning_rate=0.001)

model = tflearn.DNN(network)
model.fit(x, y, n_epoch=200, shuffle=True, validation_set=(x_test, y_test), show_metric=True, batch_size=128, run_id='cifar10_nin_dpda')

