import torch
import torch.nn as nn
import argparse


class Neural_Network(nn.Module):
	
	def __init__(self, inputSize, hiddenSize1, hiddenSize2, hiddenSize3, num_classes):
		
		super(Neural_Network, self).__init__()
		self.fc1 = nn.Linear(inputSize, hiddenSize1)
		self.relu = nn.ReLU()
		self.dropout1 = nn.Dropout(0.5)
		self.fc2 = nn.Linear(hiddenSize1, hiddenSize2)
		self.dropout2 = nn.Dropout(0.5)
		self.fc3 = nn.Linear(hiddenSize2, hiddenSize3) 
		self.dropout3 = nn.Dropout(0.5)
		self.fc4 = nn.Linear(hiddenSize3, num_classes) 
		self.sigmoid = nn.Sigmoid() 

	def forward(self, x):
		
		out = self.fc1(x)
		out = self.relu(out)
		out = self.dropout1(out)

		out = self.fc2(out)
		out = self.relu(out)
		out = self.dropout2(out)
		
		out = self.fc3(out)
		out = self.relu(out)
		out = self.dropout3(out)
		
		out_no_sigmoid = self.fc4(out)
		out = self.sigmoid(out_no_sigmoid)

		return out

def arg_parse():
	parser = argparse.ArgumentParser(description = "Semantic Features Neural Network")
	parser.add_argument('--epoch', dest='epoch', help='Number of epochs', default = 10, type=int)
	parser.add_argument('--pretrained', dest='pretrained', help='Load Pretrained Model (yes/no)', default='no', type=str)

	args = parser.parse_args()
	return args

if __name__=='__main__':

	args = arg_parse()

	n_in, n_h1, n_h2, n_h3, n_out, batch_size = 600, 512, 256, 128, 70, 10

	# model = nn.Sequential(  
	# 						nn.Linear(n_in, n_h1), 
	# 						nn.ReLU(), 
	# 						nn.Dropout(0.8),
	# 						nn.Linear(n_h1, n_h2), 
	# 						nn.ReLU(), 
	# 						nn.Dropout(0.5),
	# 						nn.Linear(n_h2, n_h3), 
	# 						nn.ReLU(), 
	# 						nn.Dropout(0.5),
	# 						nn.Linear(n_h3, n_out), 
	# 						nn.Sigmoid()  
	# 					)

	model = Neural_Network(n_in, n_h1, n_h2, n_h3, n_out)
	criterion = torch.nn.CrossEntropyLoss()

	optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, weight_decay = 0.001)

	_dir = "/home/gauravm/himangi/scene-graphs-major-project/"

	x = torch.load(_dir + "data/semantic_train_x.pt")
	y = torch.load(_dir + "data/semantic_train_y.pt")

	x_vali = torch.load(_dir + "data/semantic_validation_x.pt")
	y_vali = torch.load(_dir + "data/semantic_validation_y.pt")

	print (x.shape, y.shape)

	if args.pretrained == 'yes':
		model.load_state_dict(torch.load('./models/semantic_nn/semantic_nn_model_3h512_256_128_adam_cel_drop.pt'))

	epochs = args.epoch

	for epoch in range(epochs):
		y_pred = model(x)

		# loss = criterion(y_pred, y)
		# if criterion = cross entropy loss
		loss = criterion(y_pred, torch.max(y, 1)[1])
		print ('Loss: ', loss.item())

		optimizer.zero_grad()

		loss.backward()

		optimizer.step()

		torch.save(model.state_dict(), './models/semantic_nn/semantic_nn_model_3h512_256_128_adam_cel_drop.pt')

	# Test the model
	model.eval()
	with torch.no_grad():
		correct = 0
		total = y_vali.shape[0]
		for embedding, label in zip(x_vali, y_vali):
			output = model(torch.Tensor(embedding))
			# print output.shape
			_, predicted = torch.max(output, 0)
			correct += (predicted.item() == torch.max(label,0)[1].item())

		print (correct, total)
		print (100 * correct/total)
	




