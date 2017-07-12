import os, sys
import librosa
import pandas as pd
import pickle


dirlist = [d for d in os.listdir('./') if os.path.isdir(os.path.join('./', d))]
print dirlist

x_ = []
y_ = []
count = 0
acount = 0

for d in dirlist:
	count = count + 1
	subdir = './' + d + '/'
	audiolist = os.listdir(subdir)
	for au in audiolist:
		acount = acount + 1
		sys.stdout.write('\r%3d/%4d		at ' % (count, acount) + subdir + '	on ' + au + '		')
		sys.stdout.flush()
		y, sr = librosa.core.load(subdir + au, sr=50)
		digit = pd.cut(y, 256, labels=False) + 1
		start = len(digit) / 4
		end = (len(digit) / 4) * 3
		for idx in xrange(start, end, 50):
			x_.append(digit[idx: idx + 50])
			y_.append(d)
	acount = 0

dic = {'x_': x_, 'y_': y_}

with open('train_data', 'wb') as f:
	pickle.dump(dic, f)

print
