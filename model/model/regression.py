import torch
import torch.nn as nn


class BostonModel(nn.Module):
    def __init__(self, in_features: int, out_features: int, lr: float, epochs: int, bs: int, gamma: float):
        super().__init__()
        # 训练基础信息
        self.lr = lr
        self.epochs = epochs
        self.bs = bs
        self.gamma = gamma

        self.block = nn.Sequential(
            nn.Linear(in_features, 64), nn.ReLU(inplace=True)
        )
        self.regression = nn.Linear(64, out_features)

    def forward(self, x):
        yhat = self.block(x)
        sigma = self.regression(yhat)
        return sigma
