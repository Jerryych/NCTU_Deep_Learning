import tensorflow as tf
import numpy as np


class VGG19:

	def __init__(self):
		self.params_dic = np.load('vgg19.npy', encoding='latin1').item()
		print 'vgg19 loaded.'

	def conv_filter(self, name):
		return tf.constant(self.params_dic[name][0])

	def bias(self, name):
		return tf.constant(self.params_dic[name][1])

	def weight(self, name):
		return tf.constant(self.params_dic[name][0])

	def conv_layer(self, x, name):
		with tf.variable_scope(name):
			conv = tf.nn.conv2d(x, self.conv_filter(name), [1, 1, 1, 1], padding='SAME')
			c_biases = self.bias(name)
			return tf.nn.relu(tf.nn.bias_add(conv, c_biases))

	def fc_layer(self, x, name):
		with tf.variable_scope(name):
			shape = x.get_shape().as_list()
			dim = 1
			for d in shape[1:]:
				dim *= d
			re_x = tf.reshape(x, [-1, dim])

			fc_weights = self.weight(name)
			fc_biases = self.bias(name)

			return tf.nn.bias_add(tf.matmul(re_x, fc_weights), fc_biases)
	
	def avg_pool(self, x, name):
		return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

	def max_pool(self, x, name):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

	def build(self, x):
		print 'start building model.'

		self.conv1_1 = self.conv_layer(x, "conv1_1")
		self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
		self.pool1 = self.max_pool(self.conv1_2, 'pool1')

		self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
		self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
		self.pool2 = self.max_pool(self.conv2_2, 'pool2')

		self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
		self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
		self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
		self.conv3_4 = self.conv_layer(self.conv3_3, "conv3_4")
		self.pool3 = self.max_pool(self.conv3_4, 'pool3')

		self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
		self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
		self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
		self.conv4_4 = self.conv_layer(self.conv4_3, "conv4_4")
		self.pool4 = self.max_pool(self.conv4_4, 'pool4')

		self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
		self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
		self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
		self.conv5_4 = self.conv_layer(self.conv5_3, "conv5_4")
		self.pool5 = self.max_pool(self.conv5_4, 'pool5')

		self.fc6 = self.fc_layer(self.pool5, "fc6")
		self.relu6 = tf.nn.relu(self.fc6)

		self.fc7 = self.fc_layer(self.relu6, "fc7")
		self.relu7 = tf.nn.relu(self.fc7)

		self.fc8 = self.fc_layer(self.relu7, "fc8")
		self.prob = tf.nn.softmax(self.fc8, name="prob")

		self.data_dict = None
		print 'finishing building.'

