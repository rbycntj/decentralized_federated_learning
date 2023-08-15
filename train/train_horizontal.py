from concurrent.futures import ThreadPoolExecutor
from data.dataloader import DataLoader as DL
from data.preprocess import fashion_mnist_preprocess
from utlis.file_utls.yml_utils import read_yaml
from model.client.client_horizontal import Client
from model.model.resnet import ResNet, ResidualUnit
import matplotlib.pyplot as plt
import os

"""
    横向联邦学习：
    1.获取Fashion-mnist数据集
    2.为每个客户端划分数据集，不同客户端拥有一定数量的数据集
    3.为客户端分配邻居节点，用于梯度数据的更新
        代码中使用”环形“结构 [0->1->2->3->..->n->0] 0的邻居为1和n
    4.开始训练
        4.1 每个batch先进行正向传播与反向传播
        4.2 更新各自梯度信息
        4.3 梯度下降
"""


# 定义多线程任务
def train(client: Client):
    return client.train()


if __name__ == '__main__':
    # 1.加载配置文件并获取配置信息
    config = read_yaml('horizontal_fl')
    data_path = config['data_path']
    client_num = config['client_num']
    lr = config['lr']
    epochs = config['epochs']
    local_epochs = config['local_epochs']
    bs = config['bs']
    gamma = config['gamma']
    sample_rate = config['sample_rate']

    # 2.加载训练数据
    dl = DL(data_path)
    mnist = dl.load_horizontal_data(True, [224, 224])

    # 3.数据处理
    clients_data = fashion_mnist_preprocess(mnist, client_num, sample_rate)

    # 4.创建client_num个client
    """
    Client(client_data,net,idx)
        client_data：客户端数据集
        net：训练的网络，这里使用 resnet18 ==> Resnet(block,layers,num_classes,lr,epochs,bs,gamma)
            block：使用瓶颈结构BottleNeck还是残差单元ResidualUnit
            layers：当前结构每层有多少个block
            num_classes：分类数
            lr：学习率
            epochs：训练次数
            bs：batch_size
            gamma：SGD参数
        idx：当前client的编号
    """
    """
        若想使用resnet其他层数结构，若为深层resnet：
        1.将第一个参数残差块ResidualUnit改为瓶颈结构BottleNeck
        2.根据论文填写第二个参数层次[layer1,layer2...]
    """
    clients = [
        Client(clients_data[i], ResNet(ResidualUnit, [2, 2, 2, 2], 10, lr, epochs, local_epochs, bs, gamma), i) for i in
        range(client_num)]

    # 5.为每个client分配邻居节点，这里采用环形连接 [0->1->2->3->4->0]
    for index, client in enumerate(clients):
        pre = (index - 1 if index - 1 != -1 else len(clients) - 1)
        pst = (index + 1 if index + 1 != len(clients) else 0)
        client.neighbors = [clients[pre], clients[pst]]

    # 6.开始训练，多线程
    print("train start!")
    executor = ThreadPoolExecutor(max_workers=client_num + 1)
    tasks = [executor.submit(train, client) for client in clients]
    # 等待所有任务完成
    times = 0
    loss_list = []
    accuracy_list = []
    idx_list = []
    for task in tasks:
        times, loss, accuracy, idx = task.result()
        loss_list.append(loss)
        accuracy_list.append(accuracy)
        idx_list.append("client-" + str(idx))
    print("train end bye!")

    # 7.画图
    # 7.1 loss图
    plt.figure(1)
    x = [i for i in range(times)]
    y_list = []
    for loss in loss_list:
        y_list.append(loss)
    plt.title('loss')  # 折线图标题
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xlabel('times')  # x轴标题
    plt.ylabel('loss')  # y轴标题
    for y in y_list:
        plt.plot(x, y, marker='o', markersize=3)
    plt.legend(idx_list)  # 设置折线名称
    plt.savefig(os.getcwd() + os.sep + 'res' + os.sep + 'horizontal_loss_res_new.png')

    # 7.2 accuracy图
    plt.figure(2)
    x = [i for i in range(times)]
    y_list = []
    for accuracy in accuracy_list:
        y_list.append(accuracy)
    plt.title('accuracy')  # 折线图标题
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xlabel('times')  # x轴标题
    plt.ylabel('accuracy')  # y轴标题
    for y in y_list:
        plt.plot(x, y, marker='o', markersize=3)
    plt.legend(idx_list)  # 设置折线名称
    plt.savefig(os.getcwd() + os.sep + 'res' + os.sep + 'horizontal_accuracy_res_new.png')
