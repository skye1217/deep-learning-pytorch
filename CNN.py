#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## CNN卷积神经网络
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.optim as optim
### 改进多分类模型
# prepare dataset
batch_size=64
transform= transforms.Compose([transforms.ToTensor(), #convert image to tensor
        transforms.Normalize((0.1307,), (0.3081,))]) # 归一化,均值和方差
train_dataset=datasets.MNIST(root='/Users/yangguangqiang/Music/career-2021/deep_learning/', train=True, download=False, transform=transform)
train_loader=DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset=datasets.MNIST(root='/Users/yangguangqiang/Music/career-2021/deep_learning/', train=False, download=False, transform=transform)
test_loader=DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling=torch.nn.MaxPool2d(2)  #最大池化层
        self.FC=torch.nn.Linear(320,10)  #最后一共10类
    def forward(self,x): #多层在算y^
        batch_size=x.size(0)
        y1=F.relu(self.pooling(self.conv1(x)))
        y2=F.relu(self.pooling(self.conv2(y1)))
        y3=y2.view(batch_size,-1) #flatten: 20*4*4 to 320
        return self.FC(y3)  #最后一层没有用激活函数，后面用softmax函数算概率

model=CNN()
## 如何将模型放入 GPU 去计算: 
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss() #交叉熵-已经包含Softmax函数，不用再计算
optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.5)

def train(epoch):  #train一次的函数
    temp_loss=0
    for bat_index,data in enumerate(train_loader,0):
        x_data,y_data=data
        x_data,y_data=x_data.to(device),y_data.to(device) #和模型放到同一个显卡上
        optimizer.zero_grad()
        
        prob = model(x_data)     
        loss=criterion(prob,y_data)
        loss.backward()
        optimizer.step()
        
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












