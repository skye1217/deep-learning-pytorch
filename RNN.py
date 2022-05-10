#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## RNN 
import torch

# RNNCell ：size细节
batch_size = 1 
input_size = 4 #x的维度
hidden_size = 4  #h的维度

# RNN简单例子： "hello" -- "ohlol" 学习这个过程

idx2char = ['e', 'h', 'l', 'o'] 
x_data = [1, 0, 2, 2, 3]
y_data = [3, 1, 2, 3, 2]
one_hot_lookup = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data] #input转成vector

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size) 
y_data = torch.LongTensor(y_data).view(-1, 1)

 
class RNN(torch.nn.Module):
    def __init__(self,input_size, hidden_size, batch_size, num_layers=1):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size,
                                hidden_size=self.hidden_size,
                                num_layers=num_layers)
    def forward(self, input):
        hidden = torch.zeros(self.num_layers,
                             self.batch_size,
                             self.hidden_size) 
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)

model = RNN(input_size, hidden_size, batch_size)
criterion = torch.nn.CrossEntropyLoss() #交叉熵损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

#训练过程
for epoch in range(15):    
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs,y_data.squeeze()) 
    loss.backward()
    optimizer.step()
     
    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='') 
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))

        












