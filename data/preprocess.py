from torch.utils.data import random_split
from torch.utils.data import TensorDataset
import random
import torch

"""
    fashion-mnist数据集预处理
    主要任务：给不同客户端随机划分样本数据
"""


def fashion_mnist_preprocess(mnist, client_num: int, sample_rate: float = 0.8):
    # 为每个client的数据选取sample_rate的样本作为各自数据集，为横向联邦做准备
    clients_data = []
    # per_samples_num随机采样数据量,per_others_num剩下未被采样的数据量
    per_samples_num = int(sample_rate * len(mnist))
    per_others_num = len(mnist) - per_samples_num
    # 循环为每个客户端采样
    for i in range(client_num):
        # 采样，当前客户端最终获得per_samples_num大小的数据集
        mnist_data, other_data = random_split(mnist, [per_samples_num, per_others_num])
        clients_data.append(mnist_data)
    return clients_data


"""
    boston-data数据集预处理
"""


def boston_data_preprocess(boston_data, client_num: int, sample_rate: float = 0.8):
    samples, targets = boston_data
    # 获取张量的列数
    num_cols = samples.size(1)
    # 生成一个随机的列索引排列
    perm = torch.randperm(num_cols)
    # 按列索引重排张量
    samples = samples[:, perm]
    # 为每个client的数据选取sample_rate的样本作为各自数据集
    clients_data = []
    clients_data_tmp = []
    # 每个client分per_data个特征
    per_data = len(samples[0]) / client_num
    # 循环为每个客户端分割特征
    for i in range(client_num):
        clients_data_tmp.append((samples[:, int(i * per_data):int((i + 1) * per_data)], targets))
    # 循环为每个客户端采样
    for i in range(client_num):
        # 采样，当前客户端最终获得per_samples_num大小的数据集
        sample_index = random.sample(range(len(samples)), int(sample_rate * len(samples)))
        sample_index = sorted(sample_index)
        tmp_samples = torch.index_select(clients_data_tmp[i][0], dim=0, index=torch.IntTensor(sample_index))
        tmp_targets = torch.index_select(clients_data_tmp[i][1], dim=0, index=torch.IntTensor(sample_index))
        # 合并为dataset
        dataset = TensorDataset(tmp_samples, tmp_targets)
        clients_data.append((dataset, sample_index))  # sample_index用于纵向联邦求交
    return clients_data, int(per_data)
