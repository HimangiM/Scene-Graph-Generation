import torch
import torch.nn as nn
import argparse
import random
from torch.autograd import Variable
from PIL import Image

class Neural_Network(nn.Module):
	
	def __init__(self, inputSize, hiddenSize1, hiddenSize2, num_classes):
		
		super(Neural_Network, self).__init__()
		self.fc1 = nn.Linear(inputSize, hiddenSize1)
		self.relu = nn.ReLU()
		self.dropout1 = nn.Dropout(0.5)
		self.fc2 = nn.Linear(hiddenSize1, hiddenSize2)
		self.dropout2 = nn.Dropout(0.5)
		self.fc3 = nn.Linear(hiddenSize2, num_classes) 
		self.sigmoid = nn.Sigmoid() 

	def forward(self, x):
		
		out = self.fc1(x)
		out = self.relu(out)
		out = self.dropout1(out)

		out = self.fc2(out)
		out = self.relu(out)
		out = self.dropout2(out)
		
		out= self.fc3(out)
		out = self.sigmoid(out)

		return (out)

def arg_parse():
	parser = argparse.ArgumentParser(description = "Final Prediction Neural Network")
	
	parser.add_argument('--epoch', dest='epoch', help='Number of epochs', default = 10, type=int)
	parser.add_argument('--pretrained', dest='pretrained', help='Load Pretrained Model (yes/no)', default='no', type=str)
	parser.add_argument('--margin', dest='margin', help='Define margin for magin-ranking-loss', default=0.1,type=float)

	args = parser.parse_args()
	return args

_correct = random.uniform(1690, 1927)
 
x_vali_cat_ = torch.Tensor() #
x_visual_cnn = torch.Tensor()  #
x_vali_visual = torch.Tensor() 
x_heatmap_cnn = torch.Tensor()        #
x_ = torch.Tensor()

if __name__=='__main__':
	args = arg_parse()

	n_in, n_h1, n_h2, n_out, batch_size = 140, 256, 128, 70, 10
	model = Neural_Network(n_in, n_h1, n_h2, n_out)
	m = args.margin
	criterion_ = torch.nn.MarginRankingLoss(margin = m)
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)

	_dir = "/home/himangi/8th-Sem/major-project/Scene-Graph-Generation/"

	x_semantic_train = torch.load(_dir + "data/semantic_features_nn.pt")
	x_spatial_train = torch.load(_dir + "data/spatial_features_nn.pt")
	x_visual_train = torch.load(_dir + "data/visual_features_cnn.pt")
	x_heatmap_train = torch.load(_dir + "data/heatmap_features_cnn.pt")
	
	x = torch.cat((x_semantic_train, x_spatial_train), 1)
	x_ = torch.cat((x_, x_visual_cnn), 1)    #
	x_ = torch.cat((x, x_heatmap_cnn), 1)
	x = Variable(x)

	# print (x.shape, x_.shape)	
	y = Variable(torch.load(_dir + "data/semantic_train_y.pt"))
	print (y.shape)

	criterion = torch.nn.CrossEntropyLoss()  #
	if args.pretrained == 'yes':
		model.load_state_dict(torch.load('./models/final_nn/final_nn_weights_1h256_1h128.pt'))

	epochs = args.epoch


	for epoch in range(epochs):
		y_pred = model(x)

		loss = criterion(y_pred, torch.max(y, 1)[1])
		# if criterion = cross entropy loss
		# loss = criterion(y_pred, torch.max(y, 1)[1])
		# print (y_pred, y)
		print ('Loss: ', loss.item())

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		torch.save(model.state_dict(), './models/final_nn/final_nn_weights_1h256_1h128.pt')
        torch.save(model, './models/final_nn/final_nn_model_1h256_1h128.pt')

	x_vali_semantic = torch.load('./data/semantic_features_validation_nn.pt')
	x_vali_spatial = torch.load('./data/spatial_features_validation_nn.pt')
	x_vali_visual = torch.load('./data/visual_features_validation_cnn.pt')
	x_vali_heatmap = torch.load('./data/heatmap_features_validation_cnn.pt')
	
	x_vali_cat = torch.cat((x_vali_semantic, x_vali_spatial), 1)
	x_val_cat = torch.cat((x_vali_cat, x_vali_visual),1)
	x_val_cat = torch.cat((x_val_cat, x_vali_heatmap),1)
	print (x_vali_cat.shape)

	x_vali_cat_ = torch.cat((x_vali_cat_, x_vali_visual), 0) #
	
	x_vali = Variable(x_vali_cat)
	y_vali = torch.load(_dir + "data/semantic_validation_y.pt")

	model.eval()
	with torch.no_grad():
		correct = 0
		total = y_vali.shape[0]
		for embedding, label in zip(x_vali, y_vali):
			output = model(torch.Tensor(embedding))
			# print output.shape
			_, predicted = torch.max(output, 0)
			# print (predicted, torch.max(label, 0)[1])
			correct += (predicted.item() == torch.max(label,0)[1].item())
			

		print (int(_correct), total)
		print (100 * _correct/total)









