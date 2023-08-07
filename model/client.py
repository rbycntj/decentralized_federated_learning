import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from cryptography.hazmat.primitives.asymmetric import rsa
import random
import numpy as np
from utlis.hash_utils.full_domain_hash import hash_with_hmac


# 定义Client类
class Client():
    """
    1.client_data: tensor 客户端数据
    2.net: nn.Module 模型
    3.clients: Client 所有client信息
    4.idx: 当前client的id
    """

    def __init__(self, mnist, client_data, net: nn.Module, idx, next_client=None, neighbors=None):
        self.mnist = mnist
        self.data = client_data
        self.net = net
        self.next_client = next_client
        self.neighbors = neighbors
        self.idx = idx
        self.public_keys = {}  # 存放接收到的公钥
        self.grads = {}  # 存放接收到的权重
        self.iter = 1

        # ===================================
        # 获取自己数据集的ids
        def get_data_ids():
            data_ids = np.array(self.data)[:, 0].tolist()
            return data_ids

        # 交集数据id
        self.inter_ids = get_data_ids()  # list型
        self.inter_idx = [x for x in range(len(self.inter_ids))]

        # ===================================
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

        # 生成随机数
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
        inter_ids_origin = [x for x in self.inter_ids]
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
        print("{}-client inter_data num:{}".format(self.idx, len(self.inter_ids)))
        self.inter_idx = sorted(self.inter_idx)
        subset = torch.utils.data.Subset(self.mnist, self.inter_idx)
        self.batch_data = DataLoader(subset, batch_size=self.net.bs, shuffle=False)

    # 模型训练
    def train(self):
        batch_data = self.batch_data
        lr = self.net.lr
        epochs = self.net.epochs
        gamma = self.net.gamma

        # 检查邻居数量
        self.neighbors = list(set([neighbor for neighbor in self.neighbors if neighbor != self]))
        self.neighbors_num = len(self.neighbors)

        # 损失函数
        criterion = nn.NLLLoss()
        # 梯度下降优化器
        opt = optim.SGD(self.net.parameters(), lr=lr, momentum=gamma)

        correct = 0  # 预测正确得为0
        samples = 0  # 循环开始之前，模型一个样本都没出现过
        for epoch in range(epochs):
            for batch_idx, (x, y) in enumerate(batch_data):
                y = y.view(x.shape[0])  # 降维 y必须为1维
                x = x
                sigma = self.net.forward(x)  # 正向传播
                loss = criterion(sigma, y)
                # 反向传播得到梯度
                loss.backward()
                # 要获取邻居节点的梯度，进行fedavg操作
                # 1.1获取目前自己的梯度
                self.list_my_grads = []  # list[tensor,tensor,...]
                for my_grad in self.net.parameters():
                    self.list_my_grads.append(my_grad.grad)
                # 1.2阻塞等待其他邻居的iter次数相同
                self.wait_iter()
                # 1.3将自己的梯度发送给邻居
                self.send_grads_to_neighbor(self.neighbors)
                # 1.4等待邻居节点发送梯度
                # print("{} 号client 等待 其他节点发送梯度".format(self.idx))
                # 1.5阻塞等待
                self.wait_grad()
                # 1.6接收到权重
                # print("{} 号client 已获取 其他节点发送的梯度".format(self.idx))
                # 1.7更新梯度
                self.update_grads()
                # 梯度下降
                opt.step()
                opt.zero_grad()

                # 求解准确率，全部正确的样本数量/已经看过的样本数量
                yhat = torch.max(sigma, dim=1)[1]  # torch.max得第二个结果，也就是预测标签index
                correct += torch.sum(yhat == y)
                # 监督进度
                samples += x.shape[0]
                if (batch_idx + 1) % 125 == 0 or batch_idx == len(batch_data) - 1:
                    print("{}-client Epoch{}:[{}/{}({:.0f})%]  Loss:{:.6f},Accuracy:{:.3f}".format(
                        self.idx,
                        epoch + 1,
                        samples,
                        epochs * len(batch_data.dataset),
                        100 * samples / (epochs * len(batch_data.dataset)),
                        loss.data.item(),
                        float(100 * correct / samples)
                    ))
                self.iter += 1

    # 将梯度发送给邻居client
    def send_grads_to_neighbor(self, neighbors):
        for neighbor in neighbors:
            neighbor.receive_grads_from_neighbor(self.idx, self.list_my_grads)

    # 从邻居节点接受梯度
    def receive_grads_from_neighbor(self, idx, grads):
        self.grads[idx] = grads

    # 等待接受梯度
    def wait_grad(self):
        while True:
            if len(self.grads) != self.neighbors_num:
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
    def update_grads(self):
        my_grads = self.list_my_grads
        # 先求和
        for key, value in self.grads.items():
            # print("{}号client使用{}号邻居的梯度进行更新======".format(self.idx,key))
            for index, grad in enumerate(value):
                my_grads[index].data = my_grads[index].data + grad.data
        if self.neighbors_num != 0:
            # 再做除法
            for my_grad in my_grads:
                my_grad.data = my_grad.data / self.neighbors_num
        # 清空
        self.grads = {}
