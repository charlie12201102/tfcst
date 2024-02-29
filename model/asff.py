import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import ChannelAttention


def add_conv(in_ch, out_ch, ksize, stride):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    stage.add_module('leaky', nn.LeakyReLU(0.1))
    return stage

#自适应空间特征融合
class ASFF(nn.Module):
    def __init__(self, level, dim,features, rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = dim
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(dim[1], self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, features, 3, 1)
        elif level == 1:
            self.compress_level_0 = add_conv(dim[0], self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, features, 3, 1)
        elif level == 2:
            self.compress_level_0 = add_conv(dim[0], self.inter_dim, 1, 1)
            self.compress_level_1 = add_conv(dim[1], self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, features, 3, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, inputs):
        #首先将多尺度特征维度调整一致
        x_level_0, x_level_1, x_level_2 = inputs
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2
        
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        
        #使用softmax获取自适应权重
        levels_weight = F.softmax(levels_weight, dim=1)

        #自适应融合
        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out


#特征细化
class AFM(nn.Module):
    def __init__(self, level, dim,features, rfb=False, vis=False):
        super(AFM, self).__init__()
        self.level = level
        self.dim = dim
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = add_conv(dim[1], self.inter_dim, 3, 2)
            self.stride_level_2 = add_conv(dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, features, 3, 1)
        elif level == 1:
            self.compress_level_0 = add_conv(dim[0], self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(dim[2], self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, features, 3, 1)
        elif level == 2:
            self.compress_level_0 = add_conv(dim[0], self.inter_dim, 1, 1)
            self.compress_level_1 = add_conv(dim[1], self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, features, 3, 1)

        compress_c = 8 if rfb else 16  # when adding rfb, we use half number of channels to save memory

        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis
        self.channel_refine = ChannelAttention(self.inter_dim * 3)

    def forward(self, inputs):
        #首先将多尺度特征维度调整一致
        x_level_0, x_level_1, x_level_2 = inputs
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)

        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        #空间方向的自适应融合
        fused_out_S = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        channel_weights = self.channel_refine(torch.cat((level_0_resized, level_1_resized, level_2_resized), 1))

        # 通道方向的自适应融合
        fused_out_C = level_0_resized * channel_weights[:, 0:1, :, :] + \
                            level_1_resized * channel_weights[:, 1:2, :, :] + \
                            level_2_resized * channel_weights[:, 2:, :, :]

        fused_out = fused_out_C + fused_out_S

        out = self.expand(fused_out)

        if self.vis:
            return out, levels_weight, fused_out.sum(dim=1)
        else:
            return out