#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## pytorch-linear
import torch
x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[1], [0], [1]])
class LogisticModel(torch.nn.Module): 
    def __init__(self):
        super(LogisticModel, self).__init__() 
        self.linear = torch.nn.Linear(1, 1)
    def forward(self, x): 
        y_pred = torch.nn.functional.sigmoid(self.linear) #不同点，结果要放入sigmoid
        return y_pred

model = LogisticModel()
criterion = torch.nn.BCELoss(size_average=False) #损失函数变了
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data) 
    print(epoch, loss.item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
