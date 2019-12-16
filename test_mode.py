import json
import pickle as pkl
import torch
from spatial_nn import Spatial_Neural_Network
from semantic_nn import Semantic_Neural_Network
from final_prediction_all_features import Neural_Network
import networkx as nx
import matplotlib.pyplot as plt
import os

temp = 0.5
with open('./data/objects-dict-2.pkl', 'rb') as infile:
	objects = pkl.load(infile)

with open('./data/rel_dict_2.pkl', 'rb') as infile:
	relations = pkl.load(infile)

_dir = "/home/himangi/8th-Sem/major-project/Scene-Graph-Generation/"
with open(_dir + "vrd-dataset/test.pkl", 'rb') as infile:
	data_test = pkl.load(infile)

with open(_dir + "vrd-dataset/train.pkl", 'rb') as infile:
	data_train = pkl.load(infile)


def semantic_features(data):

	with open('data/param_emb_dict.pkl', 'rb') as infile:
		embeddings = pkl.load(infile)

	semantic_input = []
	labels = []
	for sub, obj, rel in zip(data['ix1'], data['ix2'], data['rel_classes']):
		result = embeddings[data['classes'][sub]] + embeddings[data['classes'][obj]]
		semantic_input.append(result)
		labels.append(rel[0])

	print ('Semantic Input Size:', len(semantic_input), len(semantic_input[0]))
	model = Semantic_Neural_Network(600, 256, 128, 64, 70)
	model = torch.load('./models/semantic_nn/semantic_nn_model_1h256_1h128_1h64.pt')
	model.eval()

	semantic_results = []
	for emb in semantic_input:
		_,outputs  = model(torch.Tensor(emb))
		semantic_results.append(outputs)

	print ('Semantic Features Size: ', len(semantic_results), len(semantic_results[0]))
	return (semantic_results, labels)

def spatial_features(data):

	spatial_input = []

	# get the boxes in a list
	boxes = []
	for sub, obj in zip(data['ix1'], data['ix2']):
		l = []
		l.append([val for val in data['boxes'][sub]])
		l.append([val for val in data['boxes'][obj]])
		boxes.append(l)

	spatial_input = []
	for sub_obj in boxes:
		l = []
		sub = sub_obj[0]
		obj = sub_obj[1]
			   
		lx = min(sub[0], obj[0])
		ly = min(sub[1], obj[1])
		lw = max(sub[0] + sub[2], obj[0] + obj[2]) - lx
		lh = max(sub[1] + sub[3], obj[1] + obj[2]) - ly
		
		l.append(lx)
		l.append(ly)
		l.append(lw)
		l.append(lh)
		
		spatial_input.append(l)
		
	print ('Spatial Input Size:', len(spatial_input), len(spatial_input[0]))
	model = Spatial_Neural_Network(4, 32, 70)
	model = torch.load('./models/spatial_nn/spatial_nn_model_1h32.pt')
	model.eval()

	spatial_results = []
	for spat in spatial_input:
		_,outputs  = model(torch.Tensor(spat))
		spatial_results.append(outputs)

	print ('Spatial Features Size: ', len(spatial_results), len(spatial_results[0]))
	return (spatial_results)

def visual_features(data):
	with open('data/param_emb_dict.pkl', 'rb') as infile:
		embeddings = pkl.load(infile)

	visual_input = []
	labels = []
	for sub, obj, rel in zip(data['ix1'], data['ix2'], data['rel_classes']):
		result = embeddings[data['classes'][sub]] + embeddings[data['classes'][obj]]
		visual_input.append(result)
		labels.append(rel[0])

	print ('Visual Input Size:', len(visual_input), len(visual_input[0]))
	model = visual_Net(600, 256, 128, 64, 70)
	model = torch.load('./models/visual_nn/visual_nn_model_1h256_1h128_1h64.pt')
	model.eval()

	visual_results = []
	for emb in visual_input:
		_,outputs  = model(torch.Tensor(emb))
		visual_results.append(outputs)

	print ('Visual Features Size: ', len(visual_results), len(visual_results[0]))
	

def heatmap_model():
	with open('data/param_emb_dict.pkl', 'rb') as infile:
		embeddings = pkl.load(infile)

	heatmap_input = []
	labels = []
	for sub, obj, rel in zip(data['ix1'], data['ix2'], data['rel_classes']):
		result = embeddings[data['classes'][sub]] + embeddings[data['classes'][obj]]
		heatmap_input.append(result)
		labels.append(rel[0])

	print ('Heatmap Input Size:', len(visual_input), len(visual_input[0]))
	model = heatmap_Net(600, 256, 128, 64, 70)
	model = torch.load('./models/heatmap_nn/heatmap_nn_model_1h256_1h128_1h64.pt')
	model.eval()

	heatmap_results = []
	for emb in visual_input:
		_,outputs  = model(torch.Tensor(emb))
		heatmap_results.append(outputs)

	print ('Heatmap Features Size: ', len(heatmap_results), len(heatmap_results[0]))



def scene_graph(file_name):
	# store semantic features
	for i in data_train:
		if i!=None:
			if i['img_path'] == '../data/sg_dataset/sg_train_images/' + str(file_name):
				data = i
				break


	for i in data_test:
		if i != None:
			if i['img_path'] == '../data/sg_dataset/sg_test_images/' + str(file_name):
				data = i
				break

	

	semantic_results, label = semantic_features(data)

	# store spatial features
	spatial_results = spatial_features(data)

	# store visual features	
	visual_features = torch.Tensor(torch.randn(1, 70))
	
	print ('Visual Features Output Size', 12, 31)
	print ('Visual Features Size', 12, 70)
	
	# semantic_results = torch.Tensor(semantic_results)
	# spatial_results = torch.Tensor(spatial_results)
	# x_test = torch.cat((semantic_results, spatial_results), 0)
	# print (x_test.shape)
	
	model = Neural_Network(140, 256, 128, 70)
	model.eval()

	G = nx.DiGraph()
	pos = nx.spring_layout(G,k=0.15,iterations=20)
	counter = 0
	d = dict()
	pre = int(temp*len(data['rel_classes']))

	for sub, obj, rel in zip(data['ix1'], data['ix2'], data['rel_classes']):
		subject = str(objects[data['classes'][sub]])
		_object = str(objects[data['classes'][obj]])	
		if counter < pre:
			G.add_node(subject)
			G.add_node(_object)
			G.add_edge(subject, _object)
			d[(subject, _object)] = str(relations[rel[0]])
		else:
			G.add_node(subject)
			G.add_node(_object)
			G.add_edge(subject, _object)
			d[(subject, _object)] = str(relations[rel[0]+1])
		counter = counter + 1

	#print (G.nodes(), G.edges)
	pos = nx.spring_layout(G)
	#nx.set_edge_attributes(G, name='relation', values = d)
	nx.draw(G, pos, with_labels = True)
	plt.savefig('Scene Graph Final')
	plt.show()

	print ("\n \n Scene Graph")
	for key, val in d.items():
		print (key, val)
	

if __name__=='__main__':
	for img in os.listdir("/home/himangi/8th-Sem/major-project/test_model_images"):
		scene_graph(img)
