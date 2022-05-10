#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## 对mnist数据集进行多分类
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
 
# prepare dataset
batch_size=64
transform= transforms.Compose([transforms.ToTensor(), #convert image to tensor
        transforms.Normalize((0.1307,), (0.3081,))]) # 归一化,均值和方差
train_dataset=datasets.MNIST(root='/Users/yangguangqiang/Music/career-2021/deep_learning/', train=True, download=False, transform=transform)
train_loader=DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset=datasets.MNIST(root='/Users/yangguangqiang/Music/career-2021/deep_learning/', train=False, download=False, transform=transform)
test_loader=DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

class softmaxmodel(torch.nn.Module):
    def __init__(self):
        super(softmaxmodel,self).__init__()
        self.L1=torch.nn.Linear(784,512)
        self.L2=torch.nn.Linear(512,256)
        self.L3=torch.nn.Linear(256,128)
        self.L4=torch.nn.Linear(128,64)
        self.L5=torch.nn.Linear(64,10)  #最后一共10类
    def forward(self,x): #多层在算y^
        x=x.view(-1,784)  #转换矩阵维度
        y1=F.relu(self.L1(x))
        y2=F.relu(self.L2(y1))
        y3=F.relu(self.L3(y2))
        y4=F.relu(self.L4(y3))
        return self.L5(y4)  #最后一层没有用激活函数，后面用softmax函数算概率
        
model=softmaxmodel()
criterion = torch.nn.CrossEntropyLoss() #已经包含Softmax函数，不用再计算
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.5)

def train(epoch):  #train一次的函数
    temp_loss=0
    for bat_index,data in enumerate(train_loader,0):
        x_data,y_data=data
        optimizer.zero_grad()
        
        prob = model(x_data)      #前向传播
        loss=criterion(prob,y_data)
        loss.backward()             #反馈
        optimizer.step()             #更新参数
        
        temp_loss=temp_loss+loss.item()
        if bat_index%300==299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, bat_index+1, temp_loss/300))
            temp_loss = 0.0

def test():
    correct=0
    total=0
    with torch.no_grad():  #!!test数据就不进行梯度下降了
        for data in test_loader:
            x_data,y_data=data
            prob = model(x_data) 
            _,y_pred=torch.max(prob.data,dim=1) #概率最大，属于那一类
            total=total+y_data.size(0)
            correct=correct+(y_pred==y_data).sum().item()
    print('accuracy on test set: %d %% ' % (100*correct/total))
 
 
if __name__ == '__main__':
    for epoch in range(4):
        train(epoch)
        test()
            
            
            








