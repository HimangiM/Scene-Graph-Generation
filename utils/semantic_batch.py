import json
from numpy.core import multiarray
import pickle as pkl
import torch

_dir = "/home/himangi/8th-Sem/major-project/Scene-Graph-Generation/"
def create_batch_ids():
	# Creates the batch with [subject id, object id] and [predicate id]
	with open(_dir + "vrd-dataset/train.pkl", 'rb') as infile:
		data_train = pkl.load(infile)


	x = []
	y = []
	sum_check = [] # To check whether all relationships considered or not
	none_check = 0

	for img_data in data_train:
		if img_data != None:
			# sum_check.append(len(img_data['ix1']))
			for i,j,k in zip(img_data['ix1'], img_data['ix2'], img_data['rel_classes']):
				l_x = []
				l_x.append(img_data['classes'][i])
				l_x.append(img_data['classes'][j])
				y.append(k[0])
				x.append(l_x)
		# else:
			# none_check = none_check + 1

	# print ('Number of relationships: {0}'.format(len(x)))
	return (x, y)

def create_embedding_batch(x_id, y_id):

	# Batch creation of x
	with open(_dir + 'data/param_emb_dict.pkl', 'rb') as f:
		param_dict = pkl.load(f)

	x = []
	for i in x_id:
		x.append( param_dict[x_id[0][0]] + param_dict[x_id[0][1]])

	# print (len(x), len(x[0]))
	x = torch.Tensor(x)
	torch.save(x, _dir + "data/semantic_train_x.pt")
	print (x.shape)
	# Batch creation of y
	y = []
	for i in y_id:
		l = [0 for j in range(0, 70)]
		l[i] = 1
		y.append(l)

	# print (len(y), len(y[0]))
	y = torch.Tensor(y)
	torch.save(y, _dir + 'data/semantic_train_y.pt')
	print (y.shape)

if __name__=='__main__':
	x_id, y_id = create_batch_ids()
	create_embedding_batch(x_id, y_id)
