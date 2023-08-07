import torch
import torch.nn as nn
from torch.nn import functional as F


# 模型类
class Model(nn.Module):
    def __init__(self, input_features, output_features, lr=0.15, epochs=10, bs=128, gamma=0):
        super().__init__()
        self.linear1 = nn.Linear(input_features, 128, bias=False)
        self.output = nn.Linear(128, output_features, bias=False)
        self.lr = lr
        self.epochs = epochs
        self.bs = bs
        self.gamma = gamma

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        sigma1 = torch.relu(self.linear1(x))
        sigma2 = F.log_softmax(self.output(sigma1), dim=1)
        return sigma2
