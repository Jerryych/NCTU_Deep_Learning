from audio_processor import *
import os, sys
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
                sys.stdout.write('\r%3d/%4d             at ' % (count, acount) + subdir + '     on ' + au + '           ')
                sys.stdout.flush()
               	x_.append(compute_melgram(subdir + au))
                y_.append(d)
        acount = 0

dic = {'x_': x_, 'y_': y_}

with open('image_train_data', 'wb') as f:
        pickle.dump(dic, f)

print

