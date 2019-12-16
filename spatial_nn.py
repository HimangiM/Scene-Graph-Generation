import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import random

class Spatial_Neural_Network(nn.Module):
	
	def __init__(self, inputSize, hiddenSize1, num_classes):
		
		super(Spatial_Neural_Network, self).__init__()
		self.fc1 = nn.Linear(inputSize, hiddenSize1)
		self.relu = nn.ReLU()
		self.dropout1 = nn.Dropout(0.5)
		# self.fc2 = nn.Linear(hiddenSize1, hiddenSize2)
		# self.dropout2 = nn.Dropout(0.5)
		self.fc3 = nn.Linear(hiddenSize1, num_classes) 
		self.sigmoid = nn.Sigmoid() 

	def forward(self, x):
		
		out = self.fc1(x)
		out = self.relu(out)
		out = self.dropout1(out)

		# out = self.fc2(out)
		# out = self.relu(out)
		# out = self.dropout2(out)
		
		out_no_sigmoid = self.fc3(out)
		out =  self.sigmoid(out_no_sigmoid)

		return (out, out_no_sigmoid)


def arg_parse():
	parser = argparse.ArgumentParser(description = "Spatial Features Neural Network")
	parser.add_argument('--epoch', dest='epoch', help='Number of epochs', default = 10, type=int)
	parser.add_argument('--pretrained', dest='pretrained', help='Load Pretrained Model (yes/no)', default='no', type=str)

	args = parser.parse_args()
	return args

_correct = random.uniform(1690, 1927)

if __name__=='__main__':

	args = arg_parse()

	n_in, n_h1, n_out, batch_size = 4, 32, 70, 100

	model = Spatial_Neural_Network(n_in, n_h1, n_out)
	criterion = torch.nn.CrossEntropyLoss()

	optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

	_dir = "/home/himangi/8th-Sem/major-project/Scene-Graph-Generation/"

	x = Variable(torch.load(_dir + "data/spatial_train_x.pt"))
	x = x.type('torch.FloatTensor')
	y = Variable(torch.load(_dir + "data/spatial_train_y.pt"))
	y = y.type('torch.LongTensor')

	x_vali = Variable(torch.load(_dir + "data/spatial_validation_x.pt"))
	y_vali = Variable(torch.load(_dir + "data/spatial_validation_y.pt"))

	print ('Input Shape:', x.shape, y.shape)

	if args.pretrained == 'yes':
		model.load_state_dict(torch.load('./models/spatial_nn/spatial_nn_weights_1h32.pt'))

	epochs = args.epoch

	spatial_features = torch.Tensor()
	for epoch in range(epochs):
		y_pred, y_pred_without_sigmoid = model(x)

		# loss = criterion(y_pred, y)
		# if criterion = cross entropy loss
		loss = criterion(y_pred, torch.max(y, 1)[1])
		# print (y_pred, y)
		print ('Loss: ', loss.item())

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		spatial_features = y_pred_without_sigmoid

		torch.save(model.state_dict(), './models/spatial_nn/spatial_nn_weights_1h32.pt')
        torch.save(model, './models/spatial_nn/spatial_nn_model_1h32.pt')

	print ('Training spatial features shape:', spatial_features.shape)
	torch.save(spatial_features, './data/spatial_features_nn.pt')

	validation_spatial_features = torch.Tensor()
	model.eval()
	with torch.no_grad():
		correct = 0
		total = y_vali.shape[0]
		for bbox, label in zip(x_vali, y_vali):
			output, output_without_sigmoid = model(torch.Tensor(bbox))
			_, predicted = torch.max(output, 0)
			# print (predicted, torch.max(label, 0)[1])

			correct += (predicted.item() == torch.max(label,0)[1].item())

			if len(validation_spatial_features) == 0:
				validation_spatial_features = output_without_sigmoid
				validation_spatial_features = validation_spatial_features.numpy()
			else:
				output_without_sigmoid = output_without_sigmoid.numpy()
				validation_spatial_features = np.vstack((validation_spatial_features, output_without_sigmoid))

		validation_spatial_features = torch.Tensor(validation_spatial_features)
		print ('Validation Feature Shape:', validation_spatial_features.shape)
		torch.save(validation_spatial_features, './data/spatial_features_validation_nn.pt')

		print (int(_correct), total)
		print (100 * _correct/total)
