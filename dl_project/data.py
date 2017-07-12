import numpy as np


def unpickle(file):
	import cPickle
	fo = open(file, 'rb')
	dic = cPickle.load(fo)
	dic['x_'] = np.array(dic['x_'])
	fo.close()
	if 'x_' in dic:
		dic['x_'] = dic['x_'].reshape((-1, 1, 96, 1366)).transpose([0, 2, 3, 1])
	return dic

def load_data_one(f):
	batch = unpickle(f)
	data = batch['x_']
	labels = batch['y_']
	print('Loading %s: %d' % (f, len(data)))
	return data, labels

def load_data(files, label_count):
	ln = ['blues', 'jazz', 'metal', 'classical', 'country', 'pop', 'reggae', 'disco', 'rock', 'hiphop']
	data, labels = load_data_one(files[0])
	for f in files[1:]:
		data_n, labels_n = load_data_one((f))
		data = np.append(data, data_n, axis=0)
		labels = np.append(labels, labels_n, axis=0)
	labels = np.array([[float(ln[i] == label) for i in xrange(label_count)] for label in labels])
	return data, labels
