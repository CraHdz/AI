import torch
from model.CNNNet import CNNNet

#配置运行设备为gpu或者cpu
if not torch.cuda.is_available():
    print("gpu is not avavilable, use cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


#训练模型一些常见的参数
class CNNNetConfig:
    batch_size = 100
    test_batch = 33
    momentum = 0.8
    epochs = 10
    lr = 0.01




    model = CNNNet.CNNNet()
    model_save_path = "model/CNNNet/CNNNetModel.pth"