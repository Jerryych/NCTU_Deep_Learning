import tensorflow as tf
import numpy as np
import sys
import csv


mode = sys.argv[1]
e = int(sys.argv[2])

if mode == 'y':
	exp = np.array([[10, 20], [20, 30]])
	train_length = exp[e, -1]
else:
	exp = np.array([[20, 30], [30, 50]])
	train_length = exp[e, 0]

length = exp[e, -1]
pad = length - train_length

print 'Train: ', mode
print 'Total length: ', length
print 'Training length: ', train_length
print 'Pad length: ', pad

vocab_size = 256
batch_size = 64
hidden_dim = 500
embedding_dim = 100
max_iter = 10000 + 1

padding = np.array([[vocab_size + 1 for _ in xrange(batch_size)] for _ in xrange(pad)])

enc_inp = [tf.placeholder(tf.int32, shape=(None,), name="enc_inp%i" % t) for t in xrange(length)]

labels = [tf.placeholder(tf.int32, shape=(None,), name="labels%i" % t) for t in xrange(length)]

weights = [tf.ones_like(labels_t, dtype=tf.float32) for labels_t in labels]

dec_inp = [tf.placeholder(tf.int32, shape=(None,), name="dec_inp%i" % t) for t in xrange(length)]

cell = tf.contrib.rnn.BasicLSTMCell(hidden_dim)
#cell = tf.contrib.rnn.BasicRNNCell(memory_dim)

dec_outputs, dec_memory = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(enc_inp, dec_inp, cell, vocab_size + 2, vocab_size + 2, embedding_dim)

loss = tf.contrib.legacy_seq2seq.sequence_loss(dec_outputs, labels, weights, vocab_size + 2)

learning_rate = 0.5
momentum = 0.9
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	def train_batch(batch_size):
		X = [np.random.randint(1, vocab_size + 1, train_length) for _ in xrange(batch_size)]
		X_ = [np.random.randint(0, 1, length) for _ in xrange(batch_size)]
		Y = X[:]

		X = np.array(X).T
		X_ = np.array(X_).T
		Y = np.array(Y).T

		if pad != 0:
			X = np.append(X, padding, axis=0)
			Y = np.append(Y, padding, axis=0)

		feed_dict = {enc_inp[t]: X[t] for t in xrange(length)}
		feed_dict.update({labels[t]: Y[t] for t in xrange(length)})
		feed_dict.update({dec_inp[t]: X_[t] for t in xrange(length)})

		_, loss_t = sess.run([train_op, loss], feed_dict)
		return loss_t

	def accuracy(batch_size):
		X = [np.random.randint(1, vocab_size + 1, train_length) for _ in xrange(batch_size)]
		X_ = [np.random.randint(0, 1, length) for _ in xrange(batch_size)]

		X = np.array(X).T
		X_ = np.array(X_).T

		if pad != 0:
                        Xp = np.append(X, padding, axis=0)
		else:
			Xp = X

		feed_dict = {enc_inp[t]: Xp[t] for t in xrange(length)}
		feed_dict.update({dec_inp[t]: X_[t] for t in xrange(length)})

		outputs_batch = sess.run(dec_outputs, feed_dict)
		outputs = np.array([logits_t.argmax(axis=1) for logits_t in outputs_batch])

		error_num = np.count_nonzero(X - outputs[:train_length, :])
		return float(batch_size * train_length - error_num) / float(batch_size * train_length)

	def eval(test_length):
		X_batch = [np.random.randint(1, vocab_size + 1, test_length) for _ in xrange(batch_size)]
		X_ = [np.random.randint(0, 1, length) for _ in xrange(batch_size)]

		X_batch = np.array(X_batch).T
		X_ = np.array(X_).T

		pad = length - test_length
		padding = np.array([[8 for _ in xrange(batch_size)] for _ in xrange(pad)])

		if pad != 0:
			Xp = np.append(X_batch, padding, axis=0)
		else:
			Xp = X_batch

		feed_dict = {enc_inp[t]: Xp[t] for t in xrange(length)}
		feed_dict.update({dec_inp[t]: X_[t] for t in xrange(length)})

		outputs_batch = sess.run(dec_outputs, feed_dict)
                outputs = np.array([logits_t.argmax(axis=1) for logits_t in outputs_batch])

		error_num = np.count_nonzero(X_batch - outputs[:test_length, :])
		np.savetxt(str(train_length) + '_' + str(test_length) + '_target.csv', X_batch, delimiter=',')
		np.savetxt(str(train_length) + '_' + str(test_length) + '_output.csv', outputs[:test_length, :], delimiter=',')
		return float(batch_size * test_length - error_num) / float(batch_size * test_length)

	acct = []
        losst = []

	for i in xrange(1, max_iter):
		loss_t = train_batch(batch_size)
		if i % 100 == 0:
			acc = accuracy(batch_size)
			acct.append(acc)
			losst.append(loss_t)
			print("Iter %6d, Train loss %.8f, Test acc %.8f" % (i, loss_t, acc))

	filename = str(train_length)
	out = []
        out.append(['acc_' + filename])
        for i in acct:
                out.append([i])

        with open('acc_' + filename + '.csv', 'w') as file:
                w = csv.writer(file)
                w.writerows(out)

        out = []
        out.append(['loss_' + filename])
        for i in losst:
                out.append([i])

        with open('loss_' + filename + '.csv', 'w') as file:
                w = csv.writer(file)
                w.writerows(out)

	print 'Train: ', mode
	print 'Total length: ', length
	print 'Training length: ', train_length
	print 'Pad length: ', pad
	if mode == 'y':
		test_list = exp[e]
	else:
		test_list = [exp[e][-1]]
	for i in test_list:
		print("Test length: %3d, Test acc %.8f" % (i, eval(i)))
