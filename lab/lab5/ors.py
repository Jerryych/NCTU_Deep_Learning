import tensorflow as tf
import numpy as np
import vgg19
import utils
import sys


path = sys.argv[1]
ut = utils.utils()
img = ut.load_image(path)
batch = img.reshape((1, 224, 224, 3))

with tf.Session() as sess:
	x = tf.placeholder(tf.float32, [None, 224, 224, 3])

	vgg = vgg19.VGG19()
	vgg.build(x)
	prob = sess.run(vgg.prob, feed_dict={x: batch})

	print('\nResults of %s: ' % path)
	ut.print_prob(prob[0])
