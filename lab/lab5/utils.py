import skimage
import skimage.io
import skimage.transform
import numpy as np


class utils:

	def __init__(self):
		self.file_path = 'synset.txt'
		self.synset = [l.strip() for l in open(self.file_path).readlines()]
		self.VGG_MEAN = np.array([103.939, 116.779, 123.68])
	
	# returns image of shape [224, 224, 3]
	# [height, width, depth]
	def load_image(self, path):
		# load image
		img = skimage.io.imread(path)
		# we crop image from center
		short_edge = min(img.shape[:2])
		yy = int((img.shape[0] - short_edge) / 2)
		xx = int((img.shape[1] - short_edge) / 2)
		crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
		# resize to 224, 224
		resized_img = skimage.transform.resize(crop_img, (224, 224))
		resized_img = resized_img.astype("float")
		for i in xrange(224):
			for j in xrange(224):
				resized_img[i][j][0], resized_img[i][j][2] = resized_img[i][j][2], resized_img[i][j][0]
				resized_img[i][j] = resized_img[i][j] * 255.0 - self.VGG_MEAN
		return resized_img

	# returns the top1 string
	def print_prob(self, prob):
		# print prob
		pred = np.argsort(prob)[::-1]

		# Get top1 label
		top1 = self.synset[pred[0]]
		print('Top1:')
		print('	prob %.8f | %s' %(prob[pred[0]], top1.split(' ', 1)[1]))
		# Get top5 label
		print('Top5:')	
		for i in xrange(5):
			print('	%d. prob %.8f | %s' %(i + 1, prob[pred[i]], self.synset[pred[i]].split(' ', 1)[1]))
		print('')
		return top1

	def load_image2(self, path, height=None, width=None):
		# load image
		img = skimage.io.imread(path)
		img = img / 255.0
		if height is not None and width is not None:
			ny = height
			nx = width
		elif height is not None:
			ny = height
			nx = img.shape[1] * ny / img.shape[0]
		elif width is not None:
			nx = width
			ny = img.shape[0] * nx / img.shape[1]
		else:
			ny = img.shape[0]
			nx = img.shape[1]
		return skimage.transform.resize(img, (ny, nx))


