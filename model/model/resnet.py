import torch
import torch.nn as nn
from typing import Type, Union, List, Optional  # 导入数据类型


# 3x3卷积
def conv3x3(in_channels, out_channels, stride=1, initial_zero=False):
    # 需要对bn的位置进行判断，若bn位于最后一层就初始化为0，否则不需要改变gamma值
    bn = nn.BatchNorm2d(out_channels)  # out_channels必须与输入的channels一致，bn是对每一个channel进行归一化，否则报错
    if initial_zero == True:
        nn.init.constant_(bn.weight, 0)
    # 为了复用，因为有些层后面需要加和原值x后才ReLU，因此nn.ReLU写在Sequential外面
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False),
        # stride为参数，因为位于不同地方效果不同
        # 当为残差单元第一层时，需要对原始图像的尺寸减半，因此stride=2
        # 当为瓶颈结构中间层时，保留上一层的尺寸，因此stride!=2
        bn
    )


# 1x1卷积
def conv1x1(in_channels, out_channels, stride=1, initial_zero=False):
    # 需要对bn的位置进行判断，若bn位于最后一层就初始化为0，否则不需要改变gamma值
    bn = nn.BatchNorm2d(out_channels)  # out_channels必须与输入的channels一致，bn是对每一个channel进行归一化，否则报错
    if initial_zero == True:
        nn.init.constant_(bn.weight, 0)
    # 为了复用，因为有些层后面需要加和原值x后才ReLU，因此nn.ReLU写在Sequential外面
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False),
        # stride为参数，因为位于不同地方效果不同
        # 当为残差单元第一层时，需要对原始图像的尺寸减半，因此stride=2
        # 当为瓶颈结构中间层时，保留上一层的尺寸，因此stride!=2
        bn
    )


# ResidualUnit 残差单元
class ResidualUnit(nn.Module):
    def __init__(self, out_channels: int, stride_one: int = 1, in_channels: Optional[int] = None):
        super().__init__()

        self.stride_one = stride_one
        # 当特征图尺寸需要缩小时，卷积层的输出特征图数量等于输入特征图数量的两倍
        # 当特征图尺寸不需要缩小时，卷积层的输出特征图数量等于输入特征图数量
        if stride_one != 1:
            in_channels = int(out_channels / 2)
        else:
            in_channels = out_channels

        # 拟合部分输出F(x)
        self.fit_ = nn.Sequential(
            conv3x3(in_channels, out_channels, stride=stride_one),
            nn.ReLU(inplace=True),
            conv3x3(out_channels, out_channels, initial_zero=True),
        )
        # 跳跃连接，输出x(1x1卷积之后的x)
        self.skip_conv = conv1x1(in_channels, out_channels, stride=stride_one)
        # 单独定义放在H(x)之后的ReLU
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 拟合线路
        fx = self.fit_(x)
        # 跳跃连接
        # 判断stride_one是不是等于2
        # 如果等于2那么跳跃连接部分需要添加1x1卷积控制尺寸
        # 如果等于1那么什么都不需要做
        x = x
        if self.stride_one == 2:
            x = self.skip_conv(x)
        hx = self.relu(x + fx)
        return hx


# 瓶颈
class BottleNeck(nn.Module):
    def __init__(self, middle_channels, stride_one=1, in_channels: Optional[int] = None):
        super().__init__()

        # 使用middle_channels替换out_channels
        out_channels = middle_channels * 4

        # 判断当前layers是不是紧挨着conv1x
        if in_channels == None:  # 不是则自动装配参数
            # 当第一层卷积时，特征图尺寸缩小，但通道数也缩小
            # conv2x-conv3x-conv4x-conv5x
            if stride_one != 1:
                # 当前瓶颈结构是这个layers的第一个瓶颈结构conv2x-conv3x-conv4x-conv5x
                in_channels = 2 * middle_channels
            else:
                # 当前瓶颈结构不是这个layers的第一个瓶颈结构conv2x-conv2x-conv2x
                in_channels = 4 * middle_channels

        self.fit_ = nn.Sequential(
            conv1x1(in_channels, middle_channels, stride=stride_one),
            nn.ReLU(inplace=True),
            conv3x3(middle_channels, middle_channels),
            nn.ReLU(inplace=True),
            conv1x1(middle_channels, out_channels),
        )
        self.skip_conv = conv1x1(in_channels, out_channels, stride_one)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self.fit_(x)
        x = self.skip_conv(x)
        hx = self.relu(x + fx)
        return hx


# 专门生成layers的函数
def make_layers(
        block: Type[Union[ResidualUnit, BottleNeck]],
        middle_channels: int,
        num_blocks: int,
        afterconv1: bool = False
):
    layers = []
    if afterconv1:
        layers.append(block(middle_channels, in_channels=64))
    else:
        layers.append(block(middle_channels, stride_one=2))
    for i in range(num_blocks - 1):
        layers.append(block(middle_channels))
    return layers


# 复现ResNet
class ResNet(nn.Module):
    """
    block：选择使用 残差单元ResidualUnit or 瓶颈结构BottleNeck
    layers_blocks_num：每层有多少block
    num_classes：分类数
    lr：学习率
    epochs：迭代次数
    bs：batch_size
    gamma：用于SGD
    """

    def __init__(self, block: Type[Union[ResidualUnit, BottleNeck]], layers_blocks_num: list[int], num_classes: int,
                 lr: int, epochs: int, local_epochs: int, bs: int, gamma: int):
        super().__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )
        self.layer2 = nn.Sequential(*make_layers(block, 64, layers_blocks_num[0], afterconv1=True))
        self.layer3 = nn.Sequential(*make_layers(block, 128, layers_blocks_num[1]))
        self.layer4 = nn.Sequential(*make_layers(block, 256, layers_blocks_num[2]))
        self.layer5 = nn.Sequential(*make_layers(block, 512, layers_blocks_num[3]))
        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 分类
        if block == ResidualUnit:
            self.fc = nn.Linear(512, num_classes)
        else:
            self.fc = nn.Linear(2048, num_classes)

        # 设置训练用参数
        self.lr = lr
        self.epochs = epochs
        self.local_epochs = local_epochs
        self.bs = bs
        self.gamma = gamma

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer5(self.layer4(self.layer3(self.layer2(x))))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
