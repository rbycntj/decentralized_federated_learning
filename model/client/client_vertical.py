import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from cryptography.hazmat.primitives.asymmetric import rsa
import random
from utlis.hash_utils.full_domain_hash import hash_with_hmac


# 定义Client类
class Client(object):
    """
    1.client_data: tensor 客户端数据
    2.net: nn.Module 模型
    3.idx: 当前client的id
    """

    def __init__(self, client_data, net: nn.Module, idx):
        # gpu配置
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 基础数据
        self.data = client_data[0]  # 客户端数据
        self.net = net.to(self.device)  # 模型
        self.idx = idx  # 编号

        # 邻居信息
        self.next_client = None  # 发送公钥到next_client
        self.neighbors = None  # 邻居节点
        self.neighbors_num = 0  # 邻居节点数量
        self.public_keys = {}  # 存储其他客户端公钥

        # 训练用参数
        self.sigmas = {}  # 存储邻居梯度
        self.iter = 1  # 存储当前迭代次数，保证各client同步

        # 交集数据id
        self.inter_ids = client_data[1]  # list型
        self.inter_idx = [x for x in range(len(self.inter_ids))]
        self.map = dict(zip(self.inter_ids, self.inter_idx))
        self.select_index = [x for x in range(len(self.inter_ids))]  # 求交结束后最真实的数据在data中的索引

        # 生成公钥与私钥
        # 1.1生成私钥
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        # 1.2生成公钥
        self.public_key = self.private_key.public_key()
        # 1.3各种参数
        self.d = self.private_key.private_numbers().d  # 私钥
        self.p = self.private_key.private_numbers().p
        self.q = self.private_key.private_numbers().q
        self.n = self.p * self.q
        self.e = self.private_key.private_numbers().public_numbers.e  # 公钥
        # 1.4生成随机数
        self.R = random.randint(1, self.public_key.key_size - 1)

    # 发送公钥
    def send_public_key(self):
        if self != self.next_client:
            self.next_client.receive_public_key(self.idx, self.public_key)

    # 接受公钥
    def receive_public_key(self, idx, public_key):
        self.public_keys[idx] = public_key

    # 接受数据并使用对应client的公钥加密
    def receive_data_ids(self, idx, encrypt_inter_ids):
        public_key = self.public_keys[idx]
        e = public_key.public_numbers().e
        n = public_key.public_numbers().n
        # 解密 encrypt_inter_ids
        decrypt_inter_ids = [pow(id, e, n) % n for id in encrypt_inter_ids]
        # 加密自身的inter_ids
        encrypt_my_inter_ids = [hash_with_hmac(3, pow(hash_with_hmac(2, id), e, n) % n) for id in self.inter_ids]
        return decrypt_inter_ids, encrypt_my_inter_ids

    # client向自己前一个client要数据，并求交集
    def step(self):
        # 1.初始时，获取自己client数据的id交集
        inter_ids_origin = [x for x in self.inter_ids]  # 深拷贝一份
        inter_ids = [hash_with_hmac(2, id) for id in self.inter_ids]
        # 2. Blind RSA 加密求交，向前一个client要数据并求交
        self.inter_idx = self.ask_for_data_ids(self.next_client, inter_ids)
        self.inter_ids = [inter_ids_origin[i] for i in self.inter_idx]
        return self.inter_ids

    # 计算模逆元，使用扩展欧几里得算法，时间复杂度为O(log(min(b, c)))
    def mod_inverse(self, b, c):
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            else:
                gcd, x, y = extended_gcd(b % a, a)
                return gcd, y - (b // a) * x, x

        gcd, x, _ = extended_gcd(b, c)
        if gcd == 1:
            return x % c
        else:
            return None

    # 索要数据，给其他client提供
    # inter_ids: 根据 Blind RSA 求交可知，需要将自己数据也发给对方
    def ask_for_data_ids(self, client, inter_ids):
        # 1.对随机数用私钥加密
        ciphertext = pow(self.R, self.d, self.n) % self.n
        # 2.乘以自己的数据
        pri_inter_ids = [id * ciphertext for id in inter_ids]
        # 3.传输给前一个client，返回公钥加密后的数据
        pub_inter_ids, pub_other_ids = client.receive_data_ids(self.idx, pri_inter_ids)
        # 4.对pub_inter_ids除以随机数（实际上乘以随机数的模反元素），进行全域哈希
        pub_inter_ids = [hash_with_hmac(3, id * self.mod_inverse(self.R, self.n) % self.n) for id in pub_inter_ids]
        # 5.对pub_other_ids乘以随机数
        # pub_other_ids = [id*self.R%self.n for id in pub_other_ids]
        # 6.取交集
        intersection = set(pub_inter_ids) & set(pub_other_ids)
        indices = [index for index, value in enumerate(pub_inter_ids) if value in intersection]
        return indices

    # 根据求交结果截取出训练所用的数据集
    def get_true_data(self):
        print("{}-client after_inter_data number:{}".format(self.idx, len(self.inter_ids)))
        self.select_index = [value for key, value in self.map.items() if key in self.inter_ids]
        self.select_index = sorted(self.select_index)

        # 取子集
        subset = torch.utils.data.Subset(self.data, self.select_index)
        self.batch_data = DataLoader(subset, batch_size=self.net.bs, shuffle=False)

    # 模型训练
    def train(self):
        # 训练数据
        batch_data = self.batch_data
        # 训练参数
        lr = self.net.lr
        epochs = self.net.epochs
        gamma = self.net.gamma

        # 检查邻居数量
        self.neighbors = list(set([neighbor for neighbor in self.neighbors if neighbor != self]))
        self.neighbors_num = len(self.neighbors)

        # 损失函数
        criterion = nn.MSELoss()

        # 梯度下降优化器
        opt = optim.SGD(self.net.parameters(), lr=lr, momentum=gamma)

        samples = 0  # 循环开始之前，模型一个样本都没出现过
        times = 0  # 记录次数，用于画图
        r_squared_list = []  # 记录r_squared，用于画图

        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(batch_data):
                y = y.view(x.shape[0]).to(self.device)  # 降维 y必须为1维
                x = x.to(self.device)

                # 正向传播
                sigma = self.net.forward(x)
                self.sigma_data = sigma.clone().data

                # 1.阻塞等待其他邻居的iter次数相同
                self.wait_iter()
                # 2.将自己sigma发送给邻居
                self.send_sigma_to_neighbor(self.neighbors)
                # 3.阻塞等待邻居节点发送sigma
                self.wait_sigmas()
                # 4.更新sigma
                self.update_sigma(sigma)

                # 计算损失
                loss = criterion(sigma, y.reshape(-1, 1))
                # 反向传播得到梯度
                loss.backward()

                # 梯度下降
                opt.step()
                opt.zero_grad()
                samples += x.shape[0]
                self.iter += 1

            # 监督进度
            times += 1
            r_squared = self.r_squared(y.reshape(-1, 1), sigma.reshape(-1, 1))
            r_squared_list.append(r_squared.to(torch.device('cpu')).detach().numpy())
            print("{}-client Epoch{}:[{}/{}({:.0f})%]  Loss:{:.6f},R-Squared:{:.3f}".format(
                self.idx,
                epoch + 1,
                samples,
                epochs * len(batch_data.dataset),
                100 * samples / (epochs * len(batch_data.dataset)),
                loss.data.item(),
                r_squared  # 使用r-squared判断准确率
            ))

        return times, r_squared_list, self.idx

    # R-Squared
    def r_squared(self, y_true, y_pred):
        mean_y_true = torch.mean(y_true)
        ss_total = torch.sum((y_true - mean_y_true) ** 2)
        ss_residual = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2

    # 将权重发送给邻居client
    def send_sigma_to_neighbor(self, neighbors):
        for neighbor in neighbors:
            neighbor.receive_sigma_from_neighbor(self.idx, self.sigma_data)

    # 从邻居节点接受梯度
    def receive_sigma_from_neighbor(self, idx, sigma_data):
        self.sigmas[idx] = sigma_data

    # 等待接受梯度
    def wait_sigmas(self):
        while True:
            if len(self.sigmas) != self.neighbors_num:
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
    def update_sigma(self, sigma):
        # 求和
        for key, value in self.sigmas.items():
            sigma.data += value
        sigma.data = sigma.data / (self.neighbors_num + 1)
        # 清空
        self.sigmas = {}
