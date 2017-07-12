import tensorflow as tf
import numpy as np


class VGG19_cifar10:

	def __init__(self, mode, bn):
		self.params_dic = np.load('vgg19.npy', encoding='latin1').item()
		self.mode = mode
		self.bn = bn
		print 'vgg19 loaded.'

	def conv_filter(self, shape, name):
		if self.mode == 0:
			return tf.Variable(self.params_dic[name][0], name='filter')
		else:
			return tf.Variable(tf.random_normal(shape=shape, stddev=0.03, dtype=tf.float32), name='filter')

	def bias(self, shape, std, name):
		if self.mode == 0:
			return tf.Variable(self.params_dic[name][1], name='bias')
		else:
			return tf.Variable(tf.random_normal(shape=shape, stddev=std, dtype=tf.float32), name='bias')

	def weight(self, shape, name):
		if self.mode == 0:
			return tf.Variable(self.params_dic[name][0], name='weight')
		else:
			return tf.Variable(tf.random_normal(shape=shape, stddev=0.01, dtype=tf.float32), name='weight')

	def conv_layer(self, x, shape, bn_control, name):
		with tf.variable_scope(name):
			conv = tf.nn.conv2d(x, self.conv_filter(shape, name), [1, 1, 1, 1], padding='SAME')
			c_biases = self.bias([shape[-1]], 0.03, name)
			if self.bn:
				bnc = tf.contrib.layers.batch_norm(inputs=tf.nn.bias_add(conv, c_biases), center=True, scale=True, epsilon=1e-5, updates_collections=None, is_training=bn_control)
			else:
				bnc = tf.nn.bias_add(conv, c_biases)
			return tf.nn.relu(bnc)

	def fc_layer(self, x, shape, name):
		with tf.variable_scope(name):
			s = x.get_shape().as_list()
			dim = 1
			for d in s[1:]:
				dim *= d
			re_x = tf.reshape(x, [-1, dim])

			fc_weights = self.weight(shape, name)
			fc_biases = self.bias([shape[-1]], 0.01, name)

			return tf.nn.bias_add(tf.matmul(re_x, fc_weights), fc_biases)

	def fc_layer_new(self, x, shape, name):
		with tf.variable_scope(name):
			s = x.get_shape().as_list()
			dim = 1
			for d in s[1:]:
				dim *= d
			re_x = tf.reshape(x, [-1, dim])

			fc_weights = tf.Variable(tf.random_normal(shape=shape, stddev=0.01, dtype=tf.float32), name='weight')
			fc_biases = tf.Variable(tf.random_normal(shape=[shape[-1]], stddev=0.01, dtype=tf.float32), name='bias')

			return tf.nn.bias_add(tf.matmul(re_x, fc_weights), fc_biases)
	
	def avg_pool(self, x, name):
		return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

	def max_pool(self, x, name):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

	def build(self, x, data_aug, keep_prob, bn_control):
		print 'start building model.'

		pad_image = tf.cond(data_aug, lambda: tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(img, 40, 40), x), lambda: x)
		crop_image = tf.cond(data_aug, lambda: tf.map_fn(lambda img: tf.random_crop(img, [32, 32, 3]), pad_image), lambda: pad_image)
		flip_image = tf.cond(data_aug, lambda: tf.map_fn(lambda img: tf.image.random_flip_left_right(img), crop_image), lambda: crop_image)

		self.conv1_1 = self.conv_layer(flip_image, [3, 3, 3, 64], bn_control, 'conv1_1')
		self.conv1_2 = self.conv_layer(self.conv1_1, [3, 3, 64, 64], bn_control, 'conv1_2')
		self.pool1 = self.max_pool(self.conv1_2, 'pool1')

		self.conv2_1 = self.conv_layer(self.pool1, [3, 3, 64, 128], bn_control, 'conv2_1')
		self.conv2_2 = self.conv_layer(self.conv2_1, [3, 3, 128, 128], bn_control, 'conv2_2')
		self.pool2 = self.max_pool(self.conv2_2, 'pool2')

		self.conv3_1 = self.conv_layer(self.pool2, [3, 3, 128, 256], bn_control, 'conv3_1')
		self.conv3_2 = self.conv_layer(self.conv3_1, [3, 3, 256, 256], bn_control, 'conv3_2')
		self.conv3_3 = self.conv_layer(self.conv3_2, [3, 3, 256, 256], bn_control, 'conv3_3')
		self.conv3_4 = self.conv_layer(self.conv3_3, [3, 3, 256, 256], bn_control, 'conv3_4')
		self.pool3 = self.max_pool(self.conv3_4, 'pool3')

		self.conv4_1 = self.conv_layer(self.pool3, [3, 3, 256, 512], bn_control, 'conv4_1')
		self.conv4_2 = self.conv_layer(self.conv4_1, [3, 3, 512, 512], bn_control, 'conv4_2')
		self.conv4_3 = self.conv_layer(self.conv4_2, [3, 3, 512, 512], bn_control, 'conv4_3')
		self.conv4_4 = self.conv_layer(self.conv4_3, [3, 3, 512, 512], bn_control, 'conv4_4')
		self.pool4 = self.max_pool(self.conv4_4, 'pool4')

		self.conv5_1 = self.conv_layer(self.pool4, [3, 3, 512, 512], bn_control, 'conv5_1')
		self.conv5_2 = self.conv_layer(self.conv5_1, [3, 3, 512, 512], bn_control, 'conv5_2')
		self.conv5_3 = self.conv_layer(self.conv5_2, [3, 3, 512, 512], bn_control, 'conv5_3')
		self.conv5_4 = self.conv_layer(self.conv5_3, [3, 3, 512, 512], bn_control, 'conv5_4')
		self.resh = tf.reshape(self.conv5_4, [-1, 2 * 2 * 512])

		self.fc6 = self.fc_layer_new(self.resh, [2048, 4096], 'fc6')
		self.relu6 = tf.nn.relu(self.fc6)
		self.dropout1 = tf.nn.dropout(self.relu6, keep_prob)

		self.fc7 = self.fc_layer(self.dropout1, [4096, 4096], 'fc7')
		self.relu7 = tf.nn.relu(self.fc7)
		self.dropout2 = tf.nn.dropout(self.relu7, keep_prob)

		self.fc8 = self.fc_layer_new(self.dropout2, [4096, 10], 'fc8')

		self.data_dict = None
		print 'finishing building.'

