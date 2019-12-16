import pickle as pkl
import torch
import cv2

_dir = "/home/gauravm/himangi/scene-graphs-major-project/"

def create_bb_batch():
	with open(_dir + "vrd-dataset/train.pkl", 'rb') as infile:
		data = pkl.load(infile)
	
	d = dict()

	for img_data in data:
		# print (img_data)
		if img_data != None:
			x = []
			y = []
			img_name = img_data['img_path'][8:]
			d[img_name] = []
			for i, j, k in zip(img_data['ix1'], img_data['ix2'], img_data['rel_classes']):
				l = []
				l.append([bb for bb in img_data['boxes'][i]])
				l.append([bb for bb in img_data['boxes'][j]])
				x.append(l)
				y.append(k[0])

			d[img_name].append(x)
			d[img_name].append(y)

		break

	return d

def create_image_bb_batch(d_bb):

	# Image in bounding box of x bb
	y = []
	for key, val in d_bb.items():
		# for bb in val[0]:
		print (key)	
		img = cv2.imread(_dir + "vrd-dataset/" + str(key))	
		cv2.imshow('images', img)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()		

		# One-hot vector of val[1]
		for label in val[1]:
			l = [0 for i in range(70)]
			l[label] = 1
			y.append(l)
	
	return


if __name__=='__main__':
	d_bb = create_bb_batch()
	create_image_bb_batch(d_bb)

