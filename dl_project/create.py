import numpy as np
import data, cPickle, pickle


#load data
label_names = ['blues', 'jazz', 'metal', 'classical', 'country', 'pop', 'reggae', 'disco', 'rock', 'hiphop']
label_count = len(label_names)
train_files = ['image_train_data']
x, y = data.load_data(train_files, label_count)
x = x.astype("float")
#data pre-processing
dic = {'x': x, 'y': y}
with open('pre_data', 'wb') as f:
	pickle.dump(dic, f)
	print 'dump into pre_data'
