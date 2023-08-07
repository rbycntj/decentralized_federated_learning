from concurrent.futures import ThreadPoolExecutor
from data.dataloader import DataLoader as DL
from data.preprocess import fashion_mnist_preprocess
from utlis.file_utls.yml_utils import read_yaml
from model.models import Model
from model.client import Client


def train(client: Client):
    client.train()


if __name__ == '__main__':
    # 1.加载配置文件并获取配置信息
    config = read_yaml()
    data_path = config['data_path']
    client_num = config['client_num']
    lr = config['lr']
    epochs = config['epochs']
    bs = config['bs']
    gamma = config['gamma']
    sample_rate = config['sample_rate']
    # 2.加载数据
    dl = DL(data_path)
    mnist = dl.load_data(True)
    # 3.数据处理
    clients_data = fashion_mnist_preprocess(mnist, client_num, sample_rate)
    print("=" * 6 + "create client start" + "=" * 6)
    # 4.创建client_num个client
    clients = [Client(mnist, clients_data[i], Model(len(clients_data[i][0]) - 2, 10, lr, epochs, bs, gamma), i) for i in
               range(client_num)]
    print("=" * 6 + "create client end!" + "=" * 6)
    # 公钥分发client
    for index, client in enumerate(clients):
        client.next_client = clients[(index + 1) % len(clients)]
    # 环形连接
    for index, client in enumerate(clients):
        pre = (index - 1 if index - 1 != -1 else len(clients) - 1)
        pst = (index + 1 if index + 1 != len(clients) else 0)
        client.neighbors = [clients[pre], clients[pst]]
    # 5.发送公钥
    print("=" * 6 + "send public_key start" + "=" * 6)
    for client in clients:
        client.send_public_key()
    print("=" * 6 + "send public_key end!" + "=" * 6)
    # 6.迭代求交
    print("=" * 6 + "inter data start" + "=" * 6)
    for i in range(client_num - 1):
        print("=" * 8 + "inter step {}".format(i+1) + "=" * 8)
        for client in clients:
            client.step()
    print("=" * 6 + "inter data end!" + "=" * 6)
    # 7.获取求交后数据集
    for client in clients:
        client.get_true_data()
    # 8.开始训练，多线程
    print("=" * 6 + "train start" + "=" * 6)
    executor = ThreadPoolExecutor(max_workers=client_num + 1)
    tasks = [executor.submit(train, client) for client in clients]
    # 等待所有任务完成
    for task in tasks:
        task.result()
    print("=" * 6 + "train end" + "=" * 6)
