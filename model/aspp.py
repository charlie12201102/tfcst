import torch
import torch.nn as nn
import torch.nn.functional as F

# 空间金字塔池化——拓展内容
class ASPP(nn.Module):
    def __init__(self, in_channel=512, depth=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block3 = nn.Conv2d(in_channel, depth, 3, 1, padding=3, dilation=3)
        self.atrous_block5 = nn.Conv2d(in_channel, depth, 3, 1, padding=5, dilation=5)
        self.conv_1x1_output = nn.Conv2d(depth * 4, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='nearest')

        atrous_block1 = self.atrous_block1(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block5 = self.atrous_block5(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block3, atrous_block5], dim=1))
        return net

