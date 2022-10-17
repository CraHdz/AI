import os

import matplotlib.pyplot as plt
import torch.optim
from torch import nn
import torch

from DataManger.DataLoader import DataLoder
from model import CNNNet

from torchinfo import summary
from tools import log
import cv2

from tools import tensorb
import config
from config import  CNNNetConfig



#获取训练数据,batch_size为后面SGD优化的batch_size
dataLoader = DataLoder.getInstance()
trainData = dataLoader.getTraningData(config.CNNNetConfig.batch_size)
testData = dataLoader.getTestData(config.CNNNetConfig.test_batch)



def trainModel(model, trainData, epochs, lr, device, momen, is_loginfo=False, is_loadModel=False
               ):
    print("start training model")

    if is_loadModel:
        model.load_state_dict(torch.load(config.CNNNetConfig.model_save_path))
        print("model load complete")

    #定义优化器为随机梯度下降,lr为学习率,momentum为动量因子
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momen)

    #定义损失函数
    loss_func = nn.CrossEntropyLoss()

    #将模型加载进cpu或者gpu
    model.to(device)

    #训练网络
    for e in range(epochs):
        for index, (images, labels) in enumerate(trainData):
            print(images.shape)
            print(images)
            print(labels)
            #将图片和标签加入device
            images = images.to(device)
            labels = labels.to(device)

            #计算损失函数，并将梯度清零
            out = model(images)      #通过nn.module的__call__方法去调用forward方法
            loss = loss_func(out, labels)
            optimizer.zero_grad()


            #反向传播
            loss.backward()  #对于所有需要进行梯度计算的参数w（require_grad = True）进行梯度计算,并将结果保存到w.grad中备用
            optimizer.step()  #对w值进行更新

            # tensorb.add_scalar("Loss Condition", loss, index)

            if is_loginfo:
                if index % 20 == 0:
                    log.info('epoch: {}, batch: {}, loss: {}'.format(e + 1, index + 1, loss.data))

    print("training model complete")

    #保存模型
    torch.save(model.state_dict(), config.CNNNetConfig.model_save_path)
    log.info(model.parameters())
    print("save model complete")


def testModel(model, testData, device):
    print("test DataManger geted")
    #加载模型参数
    model.load_state_dict(torch.load(config.CNNNetConfig.model_save_path))

    #开始测试
    model = model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for index, (images, labels) in enumerate(testData):
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            _, result = torch.max(out.data, 1)  #返回输入的行最大值和结果
            print(result)
            total += labels.size(0)
            print(total)
            print(result == labels)
            print((result == labels).sum())
            correct += (result == labels).sum().item()
    log.info('Accuracy: {}'.format(correct / total))

def predict(img_input, model, device):
    #加载模型
    model.load_state_dict(torch.load(config.CNNNetConfig.model_save_path))
    model.to(device)

    with torch.no_grad():
        img_input.to(device)
        out = model(img_input)

        _, pre = torch.max(out.data, 1)
        return pre.item()

def printModelInfo(model, batch_size, channel, weight, high):
    summary(model, (batch_size, channel, weight, high))

# class test():
#     def __new__(cls, *args, **kwargs):
#         print("test new is runnings")
#         return super().__new__(cls)
#
#     def __init__(self):
#         print("test init is running")
#
#     def __call__(self, *args, **kwargs):
#         print("call方法被调用了")

def main():
    print("process is running")
    # trainModel(model, trainData, epochs, lr, device, momentum)
    # comm = input("please input command:")
    # while  comm != "exit":
    #     if comm == "train":
    #         trainModel(model, trainData, epochs, lr, device, momentum)
    #     elif comm == "exit":
    #         exit(1)
    #     elif comm == "test":
    #         testModel(model, testData, device)
    #     elif comm == "predict":
    #         pass
    #     comm = input("please input command:")

    # dataLoader.savePredictImage()
    # testModel(config.CNNNetConfig.model, testData, config.device)
    trainModel(CNNNetConfig.model, trainData, CNNNetConfig.epochs, CNNNetConfig.lr, config.device, CNNNetConfig.momentum, is_loadModel=False)

    #predict image
    # img = cv2.imread("DataManger/pImage/2.png", cv2.IMREAD_GRAYSCALE)
    # img_data = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
    # result = predict(img_data, config.CNNNetConfig.model, config.device)
    # print(result)
    # tensorb.show_model(model, img_data)

# from torch_geometric.data import Data
# def testFunc():
#
#
#     edge_index = torch.tensor([[0, 1, 1, 2],
#                                [1, 0, 2, 1]], dtype=torch.long)
#     x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
#
#     data = Data(x=x, edge_index=edge_index)
#     print(data)


if __name__ == "__main__":
    main()
    # print(torch.cuda.is_available())
    #
    # x = torch.Tensor([1, 2, 3, 4, 5, 6])
    # print(x.shape)
    # pass
    # print([1] * 9)

    # print(torch.mm(a, b))
    # print(b*a)
    # print(a.size()[:-1])
    # testFunc()