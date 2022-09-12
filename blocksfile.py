import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=True)
        self.inn = nn.InstanceNorm2d(out_channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.relu(self.inn(self.conv(x)))
        return y


class DownSample(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=2):
        super(DownSample, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * stride, kernel_size, stride=stride, padding=(kernel_size-1)//2)
        self.norm = nn.InstanceNorm2d(in_channels * stride, affine=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(self.norm(self.conv(x)))
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3,7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class DoubleAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(DoubleAttentionModule, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
    def forward(self, x):
        ca = self.channel_attention(x)
        sa = self.spatial_attention(x)
        out = ca * x + sa * x
        return out


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)
    def forward(self, x):
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)
        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)
        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)
        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        # x = torch.mean(torch.cat([x0, x1, x2], dim=1), dim=1, keepdim=True)
        x = torch.cat([x0, x1, x2], dim=1)
        return x


class AINBlock(nn.Module):
    def __init__(self, channels, in_channels=1):
        super(AINBlock, self).__init__()
        self.conv_in = nn.Conv2d(in_channels, channels, kernel_size=5, stride=1, padding=2)
        self.nm_gamma = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.nm_beta = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x, feature_map):
        x = self.param_free_norm(x)
        out1 = self.relu(self.conv_in(feature_map))
        out2 = self.nm_gamma(out1)
        out3 = self.nm_beta(out2)
        out4 = x * (out2 + 1) + out3  # improvedAIN
        return out4
    def param_free_norm(self, x, epsilon=1e-5):
        x_mean = torch.mean(x, dim=[2, 3], keepdim=True)
        x_std = torch.std(x + epsilon, dim=[2, 3], keepdim=True)
        return (x - x_mean) / x_std


class AIN_ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super(AIN_ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.ain1 = AINBlock(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2)
        self.ain2 = AINBlock(channels)
        # self.relu2 = nn.LeakyReLU(0.02, inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x, feature_map):
        out1 = self.conv1(x)
        out2 = self.relu1(self.ain1(out1, feature_map))
        out3 = self.conv2(out2)
        out4 = self.ain1(out3, feature_map)
        return self.relu2(x + out4)


class ResidualBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channel_num, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x):
        y = self.relu1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return self.relu2(x + y)


class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()
        assert kernel_size % 2 == 1, 'kernel size should be odd'
        self.padding = (kernel_size - 1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size
    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return F.conv2d(x, expand_weight, None, 1, self.padding, 1, inc)


class AIN_SDResBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(AIN_SDResBlock, self).__init__()
        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)

        self.ain1 = AINBlock(channel_num)
        self.relu1 = nn.ReLU(inplace=True)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)

        self.ain2 = AINBlock(channel_num)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x, feature_map):
        out1 = self.conv1(self.pre_conv1(x))
        out2 = self.relu1(self.ain1(out1, F.interpolate(feature_map, size=[out1.size()[2], out1.size()[3]], mode='bilinear',align_corners=True)))
        out3 = self.conv2(self.pre_conv2(out2))
        out4 = self.ain2(out3, F.interpolate(feature_map, size=[out3.size()[2], out3.size()[3]], mode='bilinear',align_corners=True))
        return self.relu2(x + out4)


class SDResBlock(nn.Module):
    def __init__(self, channel_num, dilation=1, group=1):
        super(SDResBlock, self).__init__()
        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)

        self.ain1 = nn.InstanceNorm2d(channel_num, affine=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = nn.Conv2d(channel_num, channel_num, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)

        self.ain2 = nn.InstanceNorm2d(channel_num, affine=True)
        self.relu2 = nn.ReLU(inplace=True)
    def forward(self, x):
        out1 = self.conv1(self.pre_conv1(x))
        out2 = self.relu1(self.ain1(out1))
        out3 = self.conv2(self.pre_conv2(out2))
        out4 = self.ain2(out3)
        return self.relu2(x + out4)



