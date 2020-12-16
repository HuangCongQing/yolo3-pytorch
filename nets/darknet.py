import torch
import torch.nn as nn  # torch.nn重命名为nn
import math
from collections import OrderedDict

# 残差网络结构  基本的darknet块（定义了两组卷积+标准化+激活函数）
class BasicBlock(nn.Module):  # 在_make_layer函数中被调用
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()
        # 1 
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1, # 1x1卷积下降通道数
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        # 2
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3, # 3x3卷积扩张通道数
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x # 残差边

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual # 相加
        return out


class DarkNet(nn.Module):
    def __init__(self, layers): # layers=[1, 2, 8, 8, 4]
        super(DarkNet, self).__init__()
        self.inplanes = 32   # 32通道卷积
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        self.layer1 = self._make_layer([32, 64], layers[0]) # _make_layer就是残差块  32，63对应Channels，layers[0]代表堆叠次数
        self.layer2 = self._make_layer([64, 128], layers[1])
        self.layer3 = self._make_layer([128, 256], layers[2])
        self.layer4 = self._make_layer([256, 512], layers[3])
        self.layer5 = self._make_layer([512, 1024], layers[4])

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks):# 下采样+残差网络结构的堆叠
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3,
                                stride=2, padding=1, bias=False))) # 卷积
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1]))) # 标准化
        layers.append(("ds_relu", nn.LeakyReLU(0.1))) # 激活函数
        # 加入darknet模块   
        self.inplanes = planes[1]
        for i in range(0, blocks): # 残差块堆叠次数
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))  # 调用BasicBlock残差网络结构(传递参数)
        return nn.Sequential(OrderedDict(layers)) # 输出

    def forward(self, x):
        # 卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # 残差
        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    model = DarkNet([1, 2, 8, 8, 4]) # 参数数组对应layers
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained)) # 加载数据
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
