Config = \
{
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],# 九种框宽高 分别为大中小三类 每个特征层对应3个先验框
        "classes": 20, # 20类
    },
    "img_h": 416, # 图像宽高
    "img_w": 416,
}
