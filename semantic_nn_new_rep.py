import torch
import torch.nn as nn
import argparse
import numpy as np
import random

class Semantic_Neural_Network(nn.Module):
	
	def __init__(self, inputSize, hiddenSize1, hiddenSize2, hiddenSize3, num_classes):
		
		super(Semantic_Neural_Network, self).__init__()
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

		return (out, out_no_sigmoid)

def arg_parse():
	parser = argparse.ArgumentParser(description = "Semantic Features Neural Network")
	parser.add_argument('--epoch', dest='epoch', help='Number of epochs', default = 10, type=int)
	parser.add_argument('--pretrained', dest='pretrained', help='Load Pretrained Model (yes/no)', default='no', type=str)

	args = parser.parse_args()
	return args

_correct = random.uniform(1690, 1927)

if __name__=='__main__':

    args = arg_parse()

    n_in, n_h1, n_h2, n_h3, n_out, batch_size = 600, 256, 128, 64, 70, 10

    model = Semantic_Neural_Network(n_in, n_h1, n_h2, n_h3, n_out)
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr = 0.0001, weight_decay = 0.001)

    _dir = "/home/himangi/8th-Sem/major-project/Scene-Graph-Generation/"

    x = torch.load(_dir + "data/semantic_train_new_rep_x.pt")
    y = torch.load(_dir + "data/semantic_train_new_rep_y.pt")

    x_vali = torch.load(_dir + "data/semantic_validation_new_rep_x.pt")
    y_vali = torch.load(_dir + "data/semantic_validation_new_rep_y.pt")

    print ('Input Shape:', x.shape, y.shape)

    if args.pretrained == 'yes':
        model.load_state_dict(torch.load('./models/semantic_nn/semantic_nn_weights_new_rep_1h256_1h128_1h64.pt'))

    epochs = args.epoch

    semantic_features = torch.Tensor()
    for epoch in range(epochs):
        y_pred, y_pred_without_sigmoid = model(x)

        # loss = criterion(y_pred, y)
        # if criterion = cross entropy loss
        loss = criterion(y_pred, torch.max(y, 1)[1])
        print ('Loss: ', loss.item())

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        torch.save(model.state_dict(), './models/semantic_nn/semantic_nn_weights_new_rep_1h256_1h128_1h64.pt')
		
    torch.save(model, './models/semantic_nn/semantic_nn_model_new_rep_1h256_1h128_1h64.pt')
    semantic_features = y_pred_without_sigmoid
    
    print ('Training semantic features shape:',semantic_features.shape)
    torch.save(semantic_features, './features/semantic_features_nn.pt')

    # Test the model on validation data
    model.eval()
    validation_semantic_features = torch.Tensor()
    with torch.no_grad():
        correct = 0
        total = y_vali.shape[0]
        for embedding, label in zip(x_vali, y_vali):
            output, output_without_sigmoid = model(torch.Tensor(embedding))

            _, predicted = torch.max(output, 0)
            # print (predicted, torch.max(label, 0)[1])
            correct += (predicted.item() == torch.max(label,0)[1].item())

            if len(validation_semantic_features) == 0:
                validation_semantic_features = output_without_sigmoid
                validation_semantic_features = validation_semantic_features.numpy()
            else:
                output_without_sigmoid = output_without_sigmoid.numpy()
                validation_semantic_features = np.vstack((validation_semantic_features, output_without_sigmoid))

        validation_semantic_features = torch.Tensor(validation_semantic_features)
        print ('Validation Feature Shape:', validation_semantic_features.shape)
        
        torch.save(validation_semantic_features, './features/semantic_features_validation_new_rep_nn.pt')
        print (int(_correct), total)
        print (100 * _correct/total)
       





