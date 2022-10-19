import torchvision
import random
import os
import torch
from matplotlib import pyplot as plt
import cv2

train_dir = "./DataManger/train"
test_dir = "./DataManger/tests"

transforms = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
    ]
)

class DataLoder:
    instance = None
    def __new__(cls, *args, **kwargs):
        if not cls.instance:
            return super(DataLoder, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._trainSet = torchvision.datasets.MNIST(train_dir, train=True, transform=transforms, target_transform=None, download=True)
        self._testSet = torchvision.datasets.MNIST(test_dir, train=False, transform=transforms, target_transform=None, download=True)
        # self.saveImage()

    # def saveImage(self):
    #     os.mkdir('images')
    #     for i in range(len(self._train_data)):
    #         DataManger, target = next(iter(self._train_data))  # 迭代器
    #         new_data = DataManger[0][0].clone().numpy()  # 拷贝数据
    #         plt.imsave('images/' + str(i) + str(target) + '.png', new_data)
    #         print(target)

    def getTraningData(self, batch_size):
        return torch.utils.data.DataLoader(self._trainSet, batch_size=batch_size, shuffle=True)

    def getTestData(self, batch_size):
        return torch.utils.data.DataLoader(self._testSet, batch_size=batch_size, shuffle=True)


    def savePredictImage(self):
        data = self.getTraningData(1)
        for index in range(10):
            images, labels = next(iter(data))
            new_data = images[0][0].clone().numpy()
            print(new_data.shape)
            cv2.imwrite("./DataManger/pImage/" + str(index) + ".png", new_data)
            plt.imsave("./DataManger/pImage/" + str(index) + "_new.png", new_data)


    @classmethod
    def getInstance(cls):
        if not cls.instance:
            return DataLoder()
        return cls.instance




