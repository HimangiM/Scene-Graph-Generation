import pickle as pkl
import pprint
import numpy as np
import matplotlib.pyplot as plt

_dir = "/home/himangi/8th-Sem/major-project/Scene-Graph-Generation/"

with open(_dir + 'vrd-dataset/train.pkl', 'rb') as infile:
	d = pkl.load(infile)

orig_rels = []	
orig_stats = [0 for i in range(70)]
for i in range(0, len(d)):
	if d[i] != None:
		for ix1, ix2, rel in zip(d[i]['ix1'], d[i]['ix2'], d[i]['rel_classes']):
			key = (str(d[i]['classes'][ix1]), str(rel[0]), str(d[i]['classes'][ix2]))
			orig_rels.append(key)
			orig_stats[int(rel[0])] += 1

print (len(orig_rels))
clusters_dict = {}
for i in range(0, 100):
	clusters_dict[i] = []

with open(_dir + 'vrd-dataset/parsed_clusters', 'r') as infile:
	for line in infile.readlines():
		token = line.strip().split(",")
		l = []
		for t in token[:-1]:
			l.append(t.strip().split("_")[1])
		for i in l:
			for j in l:
				if j != i:
					clusters_dict[int(i)].append(j)


pprint.pprint (clusters_dict)

augmented_list = []
new_aug = []
for i in orig_rels:
	if orig_stats[int(i[1])] > 1000:
		continue
	new_aug.append(i)
	for val in clusters_dict[int(i[0])]:
		new_aug.append((val,i[1], i[2]))
	for val in clusters_dict[int(i[2])]:
		new_aug.append((i[0], i[1], val))

# print (new_aug)

aug_stats = [0 for i in range(70)]
for i in new_aug:
	# print (i[1])
	aug_stats[int(i[1])] += 1

# print (aug_stats)

bar_graph_x = []
for i, j in zip(orig_stats, aug_stats):
	bar_graph_x.append([i, j])

print (bar_graph_x)

x = [i for i in range(70)]
plt.bar(x, orig_stats)
plt.bar(x, aug_stats)
plt.show()



