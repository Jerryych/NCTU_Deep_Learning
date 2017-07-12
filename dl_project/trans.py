import tensorflow as tf
import numpy as np
from seq2seq import seq2seq
import cPickle
import pickle
import os, sys


VOCAB_SIZE = 256
DATA_SIZE = 0
BATCH_SIZE = 64
HIDDEN_UNITS = 1200
EMBEDDING = 100
MAX_EPOCH = 300
BATCH_PER_EPOCH = 0
LENGTH = 0


def accuracy(s2s, sess):
        ein = next_batch()
        din = np.array([np.random.randint(0, 1, LENGTH) for _ in xrange(BATCH_SIZE)]).T
        outputs_batch = sess.run(s2s.decoder_prediction, feed_dict={s2s.encoder_inputs: ein, s2s.decoder_inputs: din})
        print ein
        print outputs_batch
        error_num = np.count_nonzero(ein - outputs_batch)
        return float(BATCH_SIZE * LENGTH - error_num) / float(BATCH_SIZE * LENGTH)

with open('train_data', 'rb') as f:
	dic_tr = cPickle.load(f)
	xtr = dic_tr['x_']
	xtr = np.array(xtr)
	DATA_SIZE = len(xtr)
        LENGTH = len(xtr[0])
        print 'Training set loaded.'
        print 'Total:', DATA_SIZE, 'sequences.'
        print 'Sequence length:', LENGTH

with open('test_data', 'rb') as f:
        dic_ts = cPickle.load(f)
        xts = dic_ts['x_']
        xts = np.array(xts)
        DATA_SIZE = len(xts)
        LENGTH = len(xts[0])
        BATCH_PER_EPOCH = DATA_SIZE / BATCH_SIZE
        print 'Test set loaded.'
        print 'Total:', DATA_SIZE, 'sequences.'
        print 'Sequence length:', LENGTH

with tf.Session() as sess:
        s2s = seq2seq(VOCAB_SIZE, EMBEDDING, HIDDEN_UNITS)
        s2s.build()

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver()
        saver.restore(sess, 'model/model.ckpt')
        print("Model restored.")

	print 'processing training set'
	xtr_len = len(xtr)
	tr_out = []
        for i in xrange(xtr_len):
		sys.stdout.write('\r%6d/%6d' % (i + 1, xtr_len))
                sys.stdout.flush()
		temp = sess.run(s2s.encoder_final_state, feed_dict={s2s.encoder_inputs: xtr[i].reshape(LENGTH, -1)})
		tr_out.append(temp)
	print

	dic_tr['x_'] = tr_out
	with open('pre_train', 'wb') as f:
		pickle.dump(dic_tr, f)
		print 'dump pre_train'

	print 'processing test set'
	xts_len = len(xts)
	ts_out = []
	for i in xrange(xts_len):
		sys.stdout.write('\r%6d/%6d' % (i + 1, xts_len))
                sys.stdout.flush()
                temp = sess.run(s2s.encoder_final_state, feed_dict={s2s.encoder_inputs: xts[i].reshape(LENGTH, -1)})
                ts_out.append(temp)        
	print

	dic_ts['x_'] = ts_out
	with open('pre_test', 'wb') as f:
		pickle.dump(dic_ts, f)
		print 'dump pre_test'
