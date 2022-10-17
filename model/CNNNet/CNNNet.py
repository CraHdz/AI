import logging

import torch
from torch import nn


class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        #第一层包含 16个卷积核的卷积层，一个relu层，一个最大池化层
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,                    #输入图像为（1,28,28），所以输入的channel是1
                out_channels=16,                  #设置为16个卷积核，所以输出的channel是16
                kernel_size=3,                    #应该表示卷积核为3*3 也可写作（3,3）
                stride=1,
                padding=1,                        #默认填充0
            ),                                    #维度为（16,28,28）
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)           #默认步长是kernel_size，池化完后是（16,14,14）
        )


        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)         #同上，池化完成后维度为（32，7，7）
        )

        self.FC3 = nn.Linear(32*7*7, 120)
        self.FC4 = nn.Linear(120, 10)


    def forward(self, x):
        #前向传播
        a1 = self.layer1(x)
        a2 = self.layer2(a1)
        a2 = a2.view(a2.size(0), -1)
        a3 = self.FC3(a2)
        out = self.FC4(a3)
        return out

    def save(self, path):

        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))







