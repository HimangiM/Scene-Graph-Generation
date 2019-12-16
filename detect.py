import pickle as pkl
import os
import cv2
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

with open('./vrd/test.pkl', 'rb') as infile:
	data_test = pkl.load(infile)

with open('./vrd/train.pkl', 'rb') as infile:
	data_train = pkl.load(infile)

with open('objects-dict-2.pkl', 'rb') as infile:
	objects = pkl.load(infile)

l = []
d = dict()
d_classes = dict()
counter = 0
for img in os.listdir("./test_model_images"):
	print (img)
	test_image = cv2.imread('./test_model_images/' + str(img))

	for i in data_test:
		if i!=None:
			if i['img_path'] == '../data/sg_dataset/sg_test_images/' + str(img):
				data = i
				break

	for i in data_train:
		if i!=None:
			if i['img_path'] == '../data/sg_dataset/sg_train_images/' + str(img):
				data = i
				break

	#print (data)
	for val, name in zip(data['boxes'], data['classes']):
		
		print (val, objects[name])
		cv2.rectangle(test_image, (val[0], val[1]), (val[0]+val[2], val[1]+val[3]), (255, 0, 0), 2)
		x = val[0]
		h = val[1]
		cv2.putText(test_image, objects[name], (x, h), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255)) 
		
		l.append(counter)
		d[counter] = {'boxes': val, 'classes': objects[name]}
		d_classes[counter] = str(objects[name])
		counter = counter + 1
		

	cv2.imshow('im', test_image)
	cv2.imwrite('annotated_image.png', test_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	
	

	G = nx.complete_graph(len(l))
	G.add_nodes_from(l)
	nx.set_node_attributes(G, name='label', values=d)
	for d in (G.node.data()):
		print (d)
	
	print (len(G.edges()))

	print (d_classes)
	print ("Graph created and saved")
	counter = 0

	H = nx.relabel_nodes(G, d_classes, copy = True)
	nx.draw(H, with_labels = True)
	plt.savefig('scene_graph_beginning.png')
	plt.show()
	

	
	

	



