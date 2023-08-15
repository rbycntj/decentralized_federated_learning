import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from model.model.resnet import ResNet


# 定义Client类
class Client(object):
    """
    1.client_data: tensor 客户端数据
    2.net: nn.Module 模型
    3.idx: 当前client的id
    """

    def __init__(self, client_data, net: nn.Module, idx):
        # 配置gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 基础参数
        self.data = client_data  # client训练
        self.net = net.to(self.device)  # 模型
        self.idx = idx  # client编号

        # 邻居
        self.neighbors = []  # 邻居列表
        self.neighbors_num = 0  # 邻居节点数

        # 迭代训练
        self.iter = 1  # 记录迭代次数，保证各client间同步
        self.weights = {}  # 存放接收到的权重
        self.list_my_weights = []  # client自己模型的权重值列表

    """
    get_batch_data：获取batch_data
    """

    def get_batch_data(self):
        return DataLoader(
            self.data,
            self.net.bs,
            shuffle=True
        )

    """
    
    """

    # 模型训练
    def train(self):
        # 获取batch_data
        batch_data = self.get_batch_data()

        # 获取训练超参数
        lr = self.net.lr
        epochs = self.net.epochs
        local_epochs = self.net.local_epochs
        gamma = self.net.gamma

        # 检查邻居数量
        self.neighbors = list(set([neighbor for neighbor in self.neighbors if neighbor != self]))
        self.neighbors_num = len(self.neighbors)

        # 损失函数
        criterion = nn.CrossEntropyLoss()

        # 梯度下降优化器
        opt = optim.SGD(self.net.parameters(), lr=lr, momentum=gamma)

        correct = 0  # 预测正确得为0
        samples = 0  # 循环开始之前，模型一个样本都没出现过
        times = 0  # 次数，用于作图
        loss_list = []  # 损失，用于作图
        accuracy_list = []  # 准确度，用于作图

        for epoch in range(epochs):
            for local_epoch in range(local_epochs):
                for batch_idx, (x, y) in enumerate(batch_data):
                    y = y.view(x.shape[0]).to(self.device)  # 降维 y必须为1维
                    x = x.to(self.device)
                    # 正向传播
                    sigma = self.net.forward(x)
                    # 获取损失loss
                    loss = criterion(sigma, y)
                    # 反向传播得到梯度
                    loss.backward()
                    # 梯度下降
                    opt.step()
                    opt.zero_grad()

                    # 求解准确率，全部正确的样本数量/已经看过的样本数量
                    yhat = torch.max(sigma, dim=1)[1]
                    correct += torch.sum(yhat == y)
                    samples += x.shape[0]

            # 要获取邻居节点的权重，进行fedavg操作
            # 1.获取目前自己的权重
            for my_weight in self.net.parameters():
                self.list_my_weights.append(my_weight.data)
            # 2.阻塞等待其他邻居的iter次数相同
            self.wait_iter()
            # 3.将自己的权重发送给邻居
            self.send_weights_to_neighbor(self.neighbors)
            # 4.阻塞等待邻居节点发送权重
            self.wait_weights()
            # 5.更新权重
            self.update_weights(self.net.parameters())

            # 监督进度
            times += 1
            loss_list.append(loss.data.item())
            accuracy_list.append(float(100 * correct / samples))
            print("{}-client Epoch{}:[{}/{}({:.0f})%]  Loss:{:.6f},Accuracy:{:.3f}".format(
                self.idx,
                epoch + 1,
                samples,
                epochs * len(batch_data.dataset) * local_epochs,
                100 * samples / (epochs * len(batch_data.dataset) * local_epochs),
                loss.data.item(),
                float(100 * correct / samples)
            ))
            self.iter += 1
        return times, loss_list, accuracy_list, self.idx

    # 将梯度发送给邻居client
    def send_weights_to_neighbor(self, neighbors):
        for neighbor in neighbors:
            neighbor.receive_weights_from_neighbor(self.idx, self.list_my_weights)

    # 从邻居节点接受梯度
    def receive_weights_from_neighbor(self, idx, weights):
        self.weights[idx] = weights

    # 等待接受梯度
    def wait_weights(self):
        while True:
            if len(self.weights) != self.neighbors_num:
                continue
            else:
                break

    # 等待同一个iter
    def wait_iter(self):
        while True:
            is_same = True
            # 遍历所有邻居
            for neighbor in self.neighbors:
                is_same = is_same & (neighbor.iter == self.iter)
            if is_same:
                return
            else:
                continue

    # 更新梯度
    def update_weights(self, params):
        my_weights = [w for w in params]
        # 先求和
        for key, value in self.weights.items():
            for index, weight in enumerate(value):
                if my_weights[index] != None:
                    my_weights[index].data = my_weights[index].data + weight.data
        if self.neighbors_num != 0:
            # 再做除法
            for my_weight in my_weights:
                if my_weight != None:
                    my_weight.data = my_weight.data / (self.neighbors_num + 1)
        # 清空
        self.weights = {}
        self.list_my_weights = []
