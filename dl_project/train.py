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
	error_num = np.count_nonzero(ein - outputs_batch)
	return float(BATCH_SIZE * LENGTH - error_num) / float(BATCH_SIZE * LENGTH)

with open('train_data', 'rb') as f:
	dic = cPickle.load(f)
	x_ = dic['x_']
	x_ = np.array(x_)
	DATA_SIZE = len(x_)
	LENGTH = len(x_[0])
	BATCH_PER_EPOCH = DATA_SIZE / BATCH_SIZE
	print 'Training set loaded.'
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
	stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(decoder_targets, depth=VOCAB_SIZE + 2, dtype=tf.float32), logits=s2s.decoder_logits)
	loss = tf.reduce_mean(stepwise_cross_entropy)

	train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
	#train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(loss)
	#train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

	sess.run(tf.global_variables_initializer())

	saver = tf.train.Saver()

	for i in xrange(95 * BATCH_PER_EPOCH):
		#ein, dt = next_batch()
		ein = next_batch()
		din = np.array([np.random.randint(0, 1, LENGTH) for _ in xrange(BATCH_SIZE)]).T
		_, l = sess.run([train_op, loss], feed_dict={s2s.encoder_inputs: ein, s2s.decoder_inputs: din, decoder_targets: ein, learning_rate: 0.001})

		if i % BATCH_PER_EPOCH == 0:
			acc = accuracy(s2s, sess)
			print("Epoch %6d, Loss %4.8f, Acc %4.8f" % (i / BATCH_PER_EPOCH + 1, l, acc))

        for i in xrange(95 * BATCH_PER_EPOCH, 250 * BATCH_PER_EPOCH):
                #ein, dt = next_batch()
                ein = next_batch()
                din = np.array([np.random.randint(0, 1, LENGTH) for _ in xrange(BATCH_SIZE)]).T
                _, l = sess.run([train_op, loss], feed_dict={s2s.encoder_inputs: ein, s2s.decoder_inputs: din, decoder_targets: ein, learning_rate: 0.0001})

                if i % BATCH_PER_EPOCH == 0:
                        acc = accuracy(s2s, sess)
                        print("Epoch %6d, Loss %4.8f, Acc %4.8f" % (i / BATCH_PER_EPOCH + 1, l, acc))

        for i in xrange(250 * BATCH_PER_EPOCH, MAX_EPOCH * BATCH_PER_EPOCH):
                #ein, dt = next_batch()
                ein = next_batch()
                din = np.array([np.random.randint(0, 1, LENGTH) for _ in xrange(BATCH_SIZE)]).T
                _, l = sess.run([train_op, loss], feed_dict={s2s.encoder_inputs: ein, s2s.decoder_inputs: din, decoder_targets: ein, learning_rate: 0.00001})

                if i % BATCH_PER_EPOCH == 0:
                        acc = accuracy(s2s, sess)
                        print("Epoch %6d, Loss %4.8f, Acc %4.8f" % (i / BATCH_PER_EPOCH + 1, l, acc))

        '''for i in xrange(570 * BATCH_PER_EPOCH, MAX_EPOCH * BATCH_PER_EPOCH):
                #ein, dt = next_batch()
                ein = next_batch()
                din = np.array([np.random.randint(0, 1, LENGTH) for _ in xrange(BATCH_SIZE)]).T
                _, l = sess.run([train_op, loss], feed_dict={s2s.encoder_inputs: ein, s2s.decoder_inputs: din, decoder_targets: ein, learning_rate: 0.000001})

                if i % BATCH_PER_EPOCH == 0:
                        acc = accuracy(s2s, sess)
                        print("Epoch %6d, Loss %4.8f, Acc %4.8f" % (i / BATCH_PER_EPOCH + 1, l, acc))'''

	if not os.path.exists('model'):
                os.makedirs('model')

	save_path = saver.save(sess, "model/model.ckpt")
	print("Model saved in file: %s" % save_path)
