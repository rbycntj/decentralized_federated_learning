import torch
import random

"""
    fashion-mnist数据集预处理
"""


def fashion_mnist_preprocess(mnist, pieces_num, sample_rate=0.8):
    # 获取mnist数据集的样本
    mnist_data = mnist.data
    # 获取样本数量与特征数量，在这里将图片每个像素展平，每个像素点为一个特征
    samples_num = mnist_data.shape[0]
    features_num = mnist_data.shape[1] * mnist_data.shape[2]
    # 转变数据集形状为[60000,784]
    mnist_data = mnist_data.reshape(samples_num, features_num)
    # 获取样本标签 将形状变为[60000,1]
    mnist_targets = mnist.targets.reshape(samples_num, 1)
    # 深拷贝多份数据样本
    pieces_mnist_data = tuple([mnist_data.clone() for i in range(pieces_num)])
    # 首部添加唯一标识列，用于区分不同样本 尾部添加真实标签
    ids = torch.arange(1, samples_num + 1).reshape(samples_num, 1)
    pieces_data = []
    for piece_mnist_data in pieces_mnist_data:  # 循环为每个数据都加一列id
        pieces_data.append(torch.cat((torch.cat((ids, piece_mnist_data), dim=1), mnist_targets), dim=1))
    # 为每个client的数据选取4/5的样本，用于实现样本id对其
    clients_data = []
    for piece_data in pieces_data:
        # 采样
        sample_index = random.sample(range(samples_num), int(float(sample_rate) * samples_num))
        client_data = torch.index_select(piece_data, dim=0, index=torch.IntTensor(sample_index))
        clients_data.append(client_data)
    return clients_data
