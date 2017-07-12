import numpy as np


def unpickle(file):
	import cPickle
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	if 'data' in dict:
		dict['data'] = dict['data'].reshape((-1, 3, 32, 32)).transpose([0, 2, 3, 1])
	return dict

def load_data_one(f):
	batch = unpickle(f)
	data = batch['data']
	labels = batch['labels']
	print('Loading %s: %d' % (f, len(data)))
	return data, labels

def load_data(files, data_dir, label_count):
	data, labels = load_data_one(data_dir + '/' + files[0])
	for f in files[1:]:
		data_n, labels_n = load_data_one((data_dir + '/' + f))
		data = np.append(data, data_n, axis=0)
		labels = np.append(labels, labels_n, axis=0)
	labels = np.array([[float(i == label) for i in xrange(label_count)] for label in labels])
	return data, labels
