''' 
Step2: 使用特征金字塔结构提取出更有效的特征
 '''

import torch
import torch.nn as nn
from collections import OrderedDict
from nets.darknet import darknet53

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))

# 7次卷积（5次卷积+2次卷积）
def make_last_layers(filters_list, in_filters, out_filter): # out_filter=75
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1), # 1x1卷积 ，调整通道数，减少参数？
        conv2d(filters_list[0], filters_list[1], 3), # 3x3卷积进一步特征提取
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        # 2次卷积（ 3x3卷积 + 1x1卷积）
        conv2d(filters_list[0], filters_list[1], 3), # 3x3卷积
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1, # 1x1卷积， # out_filter=75
                                        stride=1, padding=0, bias=True)
    ])
    return m

class YoloBody(nn.Module):
    def __init__(self, config):
        super(YoloBody, self).__init__()
        self.config = config
        #  backbone
        self.backbone = darknet53(None) # 获取darknet结构，保存到 self.backbone

        out_filters = self.backbone.layers_out_filters # layers_out_filters = [64, 128, 256, 512, 1024]
        #  last_layer0  # 3*(5+num_classes) = 3*(5+20)=3*(4+1+20)=75
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"]) # 3*(5+num_classes) = 3*(5+20)=3*(4+1+20)=75
        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], final_out_filter0) # 7次卷积（5次卷积+2次卷积）

        #  embedding1  75
        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"]) # 75
        self.last_layer1_conv = conv2d(512, 256, 1) # 卷积
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest') # 上采样，高宽扩张为26x26
        # 26，26，256
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1) # 7次卷积（5次卷积+2次卷积）

        #  embedding2  75
        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"]) # 75
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # 52，52，128
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2) # 堆叠完成后， 7次卷积（5次卷积+2次卷积）


    def forward(self, x):
        def _branch(last_layer, layer_in): # 
            for i, e in enumerate(last_layer): # 
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in # 前5次卷积后的结果保存在out_branch
            return layer_in, out_branch
        #  backbone
        x2, x1, x0 = self.backbone(x) # 从darknet得到三个特征层
        #  yolo branch 0  （13，13，1024）
        out0, out0_branch = _branch(self.last_layer0, x0) # out0对应x0,  out0_branch对应5次卷积结果

        #  yolo branch 1
        x1_in = self.last_layer1_conv(out0_branch) # 卷积，out0_branch作为输入
        x1_in = self.last_layer1_upsample(x1_in) # 上采样
        x1_in = torch.cat([x1_in, x1], 1) # 堆叠
        out1, out1_branch = _branch(self.last_layer1, x1_in)

        #  yolo branch 2
        x2_in = self.last_layer2_conv(out1_branch) # out1_branch作为输入
        x2_in = self.last_layer2_upsample(x2_in) # 上采样
        x2_in = torch.cat([x2_in, x2], 1) # 堆叠
        out2, _ = _branch(self.last_layer2, x2_in) # “_ ”用不到就用下划线代替
        return out0, out1, out2 # 返回各层的回归预测结果

