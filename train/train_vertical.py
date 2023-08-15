from concurrent.futures import ThreadPoolExecutor
from data.dataloader import DataLoader as DL
from data.preprocess import boston_data_preprocess
from utlis.file_utls.yml_utils import read_yaml
from model.client.client_vertical import Client
from model.model.regression import BostonModel
import matplotlib.pyplot as plt
import os

"""
    纵向联邦学习：
    1.获取Boston数据集
    2.为每个客户端划分数据集的特征，每个客户端拥有不同的特征值
    3.为客户端分配邻居节点，用于wx数据的更新
        代码中使用”全连接“结构
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
    config = read_yaml('vertical_fl')
    data_path = config['data_path']
    client_num = config['client_num']
    lr = config['lr']
    epochs = config['epochs']
    bs = config['bs']
    gamma = config['gamma']
    sample_rate = config['sample_rate']

    # 2.加载数据
    dl = DL(data_path)
    samples, targets = dl.load_vertical_data()

    # 3.数据处理
    clients_data, per_data = boston_data_preprocess((samples, targets), client_num, sample_rate)

    # 4.创建client_num个client
    """
        Client(client_data,net,idx)
            client_data：客户端数据集
            net：训练的网络，这里使用 regression ==> BostonModel(in_features,out_features,lr,epochs,bs,gamma)
            idx：当前client的编号
        """
    clients = [
        Client(clients_data[i], BostonModel(per_data, 1, lr, epochs, bs, gamma), i) for i in
        range(client_num)]
    # 配置公钥分发给哪个client 这里分发给下一个客户端 [0->1->2->3->..->n->0]
    for index, client in enumerate(clients):
        client.next_client = clients[(index + 1) % len(clients)]
    # 全连接
    for client in clients:
        client.neighbors = [c for c in clients if c != client]

    # 5.发送公钥
    for client in clients:
        client.send_public_key()

    # 6.迭代求交
    for i in range(client_num - 1):
        for client in clients:
            client.step()

    # 7.获取求交后数据集
    for client in clients:
        client.get_true_data()

    # 8.开始训练，多线程w
    print("train start!")
    executor = ThreadPoolExecutor(max_workers=client_num + 1)
    tasks = [executor.submit(train, client) for client in clients]
    # 等待所有任务完成
    r_squared_list = []
    idx_list = []
    times = 0
    for task in tasks:
        times, r_squared, idx = task.result()
        r_squared_list.append(r_squared)
        idx_list.append('client-' + str(idx))
        times = times
    print("train end bye!")

    # 9.画图
    x = [i for i in range(times)]
    y_list = []
    for r_squared in r_squared_list:
        y_list.append(r_squared)
    plt.title('R-Squared')  # 折线图标题
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.xlabel('times')  # x轴标题
    plt.ylabel('r_squared')  # y轴标题
    for y in y_list:
        plt.plot(x, y, marker='o', markersize=3)
    plt.legend(idx_list)  # 设置折线名称
    plt.savefig(os.getcwd() + os.sep + 'res' + os.sep + 'vertical_res_new.png')
