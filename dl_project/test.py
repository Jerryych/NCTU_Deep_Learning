import tensorflow as tf
import numpy as np
from seq2seq import seq2seq
import cPickle
import os


VOCAB_SIZE = 256
DATA_SIZE = 0
BATCH_SIZE = 64
HIDDEN_UNITS = 1200
EMBEDDING = 100
MAX_EPOCH = 300
BATCH_PER_EPOCH = 0
LENGTH = 0


def next_batch():
	get = np.random.choice(DATA_SIZE, BATCH_SIZE, False)
        xs = x_[get]
	#ei = xs[:, :200]
	#dt = xs[:, 200:]
	#return ei.T, dt.T
	return xs.T

def accuracy(s2s, sess):
	ein = next_batch()
	din = np.array([np.random.randint(0, 1, LENGTH) for _ in xrange(BATCH_SIZE)]).T
	outputs_batch = sess.run(s2s.decoder_prediction, feed_dict={s2s.encoder_inputs: ein, s2s.decoder_inputs: din})
	print ein
	print outputs_batch
	error_num = np.count_nonzero(ein - outputs_batch)
	return float(BATCH_SIZE * LENGTH - error_num) / float(BATCH_SIZE * LENGTH)

with open('test_data', 'rb') as f:
	dic = cPickle.load(f)
	x_ = dic['x_']
	x_ = np.array(x_)
	DATA_SIZE = len(x_)
	LENGTH = len(x_[0])
	BATCH_PER_EPOCH = DATA_SIZE / BATCH_SIZE
	print 'Test set loaded.'
	print 'Total:', DATA_SIZE, 'sequences.'
	print 'Sequence length:', LENGTH

with tf.Session() as sess:
	print 'Epoches:', MAX_EPOCH
	print 'Batch size:', BATCH_SIZE
	print 'Batch per epoch:', BATCH_PER_EPOCH
	print 'Number of iteration:', MAX_EPOCH * BATCH_PER_EPOCH
	s2s = seq2seq(VOCAB_SIZE, EMBEDDING, HIDDEN_UNITS)
	s2s.build()

	decoder_targets = tf.placeholder(shape=[None, None], dtype=tf.int32, name='decoder_targets')
	learning_rate = tf.placeholder(tf.float32)

	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()
	saver.restore(sess, 'model/model.ckpt')
	print("Model restored.")

	for i in xrange(BATCH_PER_EPOCH):
		acc = accuracy(s2s, sess)
		print("Iter %6d, Test acc %4.8f" % (i, acc))
