import numpy as np
import data, cPickle, pickle


mean = np.array([123.68, 116.779, 103.939])
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
			x[i][j][k] = x[i][j][k] - mean
for i in xrange(len(x_test)):
	for j in xrange(32):
		for k in xrange(32):
			x_test[i][j][k] = x_test[i][j][k] - mean
dic = {'train': {'x': x, 'y': y}, 'test': {'x': x_test, 'y': y_test}}
with open('pre_data_rgb', 'wb') as f:
	pickle.dump(dic, f)

for i in xrange(len(x)):
        for j in xrange(32):
                for k in xrange(32):
                        x[i][j][k][0], x[i][j][k][2] = x[i][j][k][2], x[i][j][k][0]
for i in xrange(len(x_test)):
        for j in xrange(32):
                for k in xrange(32):
			x_test[i][j][k][0], x_test[i][j][k][2] = x_test[i][j][k][2], x_test[i][j][k][0]
dic = {'train': {'x': x, 'y': y}, 'test': {'x': x_test, 'y': y_test}}
with open('pre_data_bgr', 'wb') as f:
	pickle.dump(dic, f)
