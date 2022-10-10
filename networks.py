import torch
import torch.nn as nn
from blocksfile import ResidualBlock, DownSample, AIN_SDResBlock, Get_gradient_nopadding,\
    DoubleAttentionModule, AIN_ResBlock, Conv1x1
import torch.nn.functional as F


class TextureNet(nn.Module):
    def __init__(self, nf):
        super(TextureNet, self).__init__()
        self.encoder_res1 = ResidualBlock(nf)

        self.encoder_conv1 = nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=True)
        self.encoder_relu1 = nn.ReLU(inplace=True)
        self.encoder_res2 = ResidualBlock(nf * 2)

        self.encoder_conv2 = nn.Conv2d(nf * 2, nf * 4, 3, 2, 1, bias=True)
        self.encoder_relu2 = nn.ReLU(inplace=True)
        self.encoder_res3 = ResidualBlock(nf * 4)

        self.decoder_conv1 = nn.Conv2d(nf * 4, nf * 2, 1, 1, 0, bias=True)
        self.decoder_relu1 = nn.ReLU(inplace=True)
        self.decoder_res1 = ResidualBlock(nf * 2)

        self.decoder_conv2 = nn.Conv2d(nf * 2, nf, 1, 1, 0, bias=True)
        self.decoder_relu2 = nn.ReLU(inplace=True)
        self.decoder_res2 = ResidualBlock(nf)
    def forward(self, x):
        out1 = self.encoder_res1(x)

        out2 = self.encoder_relu1(self.encoder_conv1(out1))
        out3 = self.encoder_res2(out2)

        out4 = self.encoder_relu2(self.encoder_conv2(out3))
        out5 = self.encoder_res3(out4)

        out6 = F.interpolate(out5, size=[out3.size()[2], out3.size()[3]], mode='bilinear')
        out7 = self.decoder_relu1(self.decoder_conv1(out6))
        out8 = self.decoder_res1(out7 + out3)

        out9 = F.interpolate(out8, size=[out1.size()[2], out1.size()[3]], mode='bilinear')
        out10 = self.decoder_relu2(self.decoder_conv2(out9))
        out11 = self.decoder_res2(out10 + out1)

        return out11


class LightNet(nn.Module):
    def __init__(self, nf):
        super(LightNet, self).__init__()
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.in2 = nn.InstanceNorm2d(nf, affine=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=True)
        self.in3 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)
        self.in4 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(nf * 2, nf * 4, 3, 2, 1, bias=True)
        self.in5 = nn.InstanceNorm2d(nf * 4, affine=True)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=True)
        self.in6 = nn.InstanceNorm2d(nf * 4, affine=True)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(nf * 4, nf * 2, 1, 1, 0, bias=True)
        self.in7 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)
        self.in8 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.relu8 = nn.ReLU(inplace=True)

        self.conv9 = nn.Conv2d(nf * 2, nf, 1, 1, 0, bias=True)
        self.in9 = nn.InstanceNorm2d(nf, affine=True)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.in10 = nn.InstanceNorm2d(nf, affine=True)
        self.relu10 = nn.ReLU(inplace=True)
    def forward(self, x):
        out2 = self.relu2(self.in2(self.conv2(x)))

        out3 = self.relu3(self.in3(self.conv3(out2)))
        out4 = self.relu4(self.in4(self.conv4(out3)))

        out5 = self.relu5(self.in5(self.conv5(out4)))
        out6 = self.relu6(self.in6(self.conv6(out5)))

        up1 = F.interpolate(out6, size=[out4.size()[2], out4.size()[3]], mode='bilinear')
        out7 = self.relu7(self.in7(self.conv7(up1)))
        out8 = self.relu8(self.in8(self.conv8(out7 + out4)))

        up2 = F.interpolate(out8, size=[out2.size()[2], out2.size()[3]], mode='bilinear')
        out9 = self.relu9(self.in9(self.conv9(up2)))
        out10 = self.relu10(self.in10(self.conv10(out9 + out2)))

        return out10


class LightenNet(nn.Module):
    def __init__(self, in_channels, out_channels, nf, kernel_size=3, stride=1, padding=1, bias=False):
        super(LightenNet, self).__init__()
        # ----------------------------------------- Encoder ---------------------------------------- #
        self.conv_in = nn.Conv2d(in_channels, nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.relu_in = nn.ReLU()
        self.Encoder_AINResBlock1 = AIN_ResBlock(nf)
        self.Encoder_AINResBlock2 = AIN_ResBlock(nf)

        self.Encoder_DownSample1 = DownSample(nf)
        self.Encoder_AINResBlock3 = AIN_ResBlock(nf * 2)
        self.Encoder_AINResBlock4 = AIN_ResBlock(nf * 2)
        self.Encoder_DownSample2 = DownSample(nf * 2)

        self.Encoder_AINSDResBlock1 = AIN_SDResBlock(nf * 4, 2)
        self.Encoder_AINSDResBlock2 = AIN_SDResBlock(nf * 4, 2)
        self.Encoder_AINSDResBlock3 = AIN_SDResBlock(nf * 4, 4)
        self.Encoder_AINSDResBlock4 = AIN_SDResBlock(nf * 4, 4)
        self.Encoder_AINSDResBlock5 = AIN_SDResBlock(nf * 4, 8)
        self.Encoder_AINSDResBlock6 = AIN_SDResBlock(nf * 4, 8)
        # ------------------------------------------------------------------------------------------ #

        # ------------------------------------- Texture Branch ------------------------------------- #
        self.Texture_input = Get_gradient_nopadding()
        self.Texture_conv_in = nn.Conv2d(in_channels, nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.Texture_relu_in = nn.ReLU()
        self.Texture_dam1 = DoubleAttentionModule(nf)
        self.Texture_dam2 = DoubleAttentionModule(nf * 2)
        self.Texture_dam3 = DoubleAttentionModule(nf * 4)
        self.Texture_dam2_conv = nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.Texture_dam3_conv = nn.Conv2d(nf * 4, nf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.Texture_dam_add_conv = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=bias)
        self.Texture_dam2_norm = nn.InstanceNorm2d(nf, affine=True)
        self.Texture_dam3_norm = nn.InstanceNorm2d(nf, affine=True)
        self.Texture_dam_add_norm = nn.InstanceNorm2d(nf, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.Texture_net = TextureNet(nf)
        self.Texture_conv_out = nn.Conv2d(nf, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.Texture_sigmoid = nn.Sigmoid()
        # ------------------------------------------------------------------------------------------ #

        # ------------------------------------- Light   Branch ------------------------------------- #
        self.Light_conv_in = nn.Conv2d(1, nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.Light_relu_in = nn.ReLU()
        self.Light_net = LightNet(nf)
        self.Light_conv_out = nn.Conv2d(nf, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.Light_sigmoid = nn.Sigmoid()
        # ------------------------------------------------------------------------------------------ #

        # ----------------------------------------- Decoder ---------------------------------------- #
        self.Decoder_conv1x1_1 = Conv1x1(nf * 4, nf * 2)

        self.Decoder_texture_downsample = DownSample(nf)
        self.Decoder_tadd_conv1 = nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1, bias=bias)
        self.Decoder_tadd_norm1 = nn.InstanceNorm2d(nf * 2, affine=True)
        self.Decoder_maxpool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.Decoder_maxpool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
        self.Decoder_ainresblock1 = AIN_ResBlock(nf * 2)
        self.Decoder_ainresblock2 = AIN_ResBlock(nf * 2)

        self.Decoder_conv1x1_2 = Conv1x1(nf * 2, nf)

        self.Decoder_tadd_conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.Decoder_tadd_norm2 = nn.InstanceNorm2d(nf, affine=True)
        self.Decoder_ainresblock3 = AIN_ResBlock(nf)
        self.Decoder_ainresblock4 = AIN_ResBlock(nf)
        self.Decoder_conv_out = nn.Conv2d(nf, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.Decoder_sigmoid = nn.Sigmoid()
        # ------------------------------------------------------------------------------------------ #

    def forward(self, x):
        # light
        l1 = torch.clamp(torch.max(x, dim=1)[0].unsqueeze(1), min=0.0, max=1.0)
        l2 = self.Light_relu_in(self.Light_conv_in(l1))
        l3 = self.Light_net(l2)
        l4 = self.Light_conv_out(l3)
        lout = self.Light_sigmoid(l4 + l1)

        # encoder
        e1 = self.relu_in(self.conv_in(x))
        e2 = self.Encoder_AINResBlock1(e1, l4)
        e3 = self.Encoder_AINResBlock2(e2, l4)
        e4 = self.Encoder_DownSample1(e3)
        e5 = self.Encoder_AINResBlock3(e4, F.interpolate(l4, size=[e4.size()[2], e4.size()[3]], mode='bilinear', align_corners=True))
        e6 = self.Encoder_AINResBlock4(e5, F.interpolate(l4, size=[e4.size()[2], e4.size()[3]], mode='bilinear', align_corners=True))
        e7 = self.Encoder_DownSample2(e6)
        e8 = self.Encoder_AINSDResBlock1(e7, l4)
        e9 = self.Encoder_AINSDResBlock2(e8, l4)
        e10 = self.Encoder_AINSDResBlock3(e9, l4)
        e11 = self.Encoder_AINSDResBlock4(e10, l4)
        e12 = self.Encoder_AINSDResBlock5(e11, l4)
        e13 = self.Encoder_AINSDResBlock6(e12, l4)

        # gradient
        t1 = torch.clamp(self.Texture_input(x), min=0.0, max=1.0)
        t2 = self.Texture_relu_in(self.Texture_conv_in(t1))
        t3 = self.Texture_dam1(e1)
        t4 = self.Texture_dam2(e4)
        t5 = self.Texture_dam3(e7)
        ta1 = F.interpolate(t4, size=[t3.size()[2], t3.size()[3]], mode='bilinear', align_corners=True)
        ta2 = F.interpolate(t5, size=[t3.size()[2], t3.size()[3]], mode='bilinear', align_corners=True)
        toa1 = self.relu(self.Texture_dam2_norm(self.Texture_dam2_conv(ta1)))
        toa2 = self.relu(self.Texture_dam3_norm(self.Texture_dam3_conv(ta2)))
        t6 = self.relu(self.Texture_dam_add_norm(self.Texture_dam_add_conv(t3 + toa1 + toa2)))
        t7 = self.Texture_net(t2 + t6)
        t8 = self.Texture_conv_out(t7)
        tout = self.Texture_sigmoid(t8)

        # decoder
        d1 = self.Decoder_conv1x1_1(F.interpolate(e13, size=[e4.size()[2], e4.size()[3]], mode='bilinear', align_corners=True))
        d2 = self.Decoder_texture_downsample(t7)
        d3 = self.relu(self.Decoder_tadd_norm1(self.Decoder_tadd_conv1(d2 + d1)))
        d4 = self.Decoder_ainresblock1(d3, F.interpolate(l4, size=[e4.size()[2], e4.size()[3]], mode='bilinear', align_corners=True))
        d5 = self.Decoder_ainresblock2(d4, F.interpolate(l4, size=[e4.size()[2], e4.size()[3]], mode='bilinear', align_corners=True))
        d6 = self.Decoder_conv1x1_2(F.interpolate(d5, size=[e3.size()[2], e3.size()[3]], mode='bilinear', align_corners=True))
        d7 = self.relu(self.Decoder_tadd_norm2(self.Decoder_tadd_conv2(d6 + t7)))
        d8 = self.Decoder_ainresblock3(d7, l4)
        d9 = self.Decoder_ainresblock4(d8, l4)
        d10 = self.Decoder_conv_out(d9)
        dout = self.Decoder_sigmoid(d10 + x)

        return t1, tout, l1, lout, dout


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()

#
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# net = LightenNet(3, 3, 32)
# net.to(device)
#
# num_params = 0
# for param in net.parameters():
#     num_params += param.numel()
# # print(net)
# print('Total number of parameters: %d' % num_params)
#
# total = sum([param.nelement() for param in net.parameters()])
# print(total / 1e6)
# from torchsummary import summary
# summary(net, input_size=(3, 251, 251))








# import torch
# import torch.nn as nn
# from blocksfile import ResidualBlock, DownSample, AIN_SDResBlock, Get_gradient_nopadding,\
#     DoubleAttentionModule, AIN_ResBlock, Conv1x1
# import torch.nn.functional as F
#
#
# class TextureNet(nn.Module):
#     def __init__(self, nf):
#         super(TextureNet, self).__init__()
#         self.encoder_res1 = ResidualBlock(nf)
#
#         self.encoder_conv1 = nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=True)
#         self.encoder_relu1 = nn.ReLU(inplace=True)
#         self.encoder_res2 = ResidualBlock(nf * 2)
#
#         self.encoder_conv2 = nn.Conv2d(nf * 2, nf * 4, 3, 2, 1, bias=True)
#         self.encoder_relu2 = nn.ReLU(inplace=True)
#         self.encoder_res3 = ResidualBlock(nf * 4)
#
#         self.decoder_conv1 = nn.Conv2d(nf * 4, nf * 2, 1, 1, 0, bias=True)
#         self.decoder_relu1 = nn.ReLU(inplace=True)
#         self.decoder_res1 = ResidualBlock(nf * 2)
#
#         self.decoder_conv2 = nn.Conv2d(nf * 2, nf, 1, 1, 0, bias=True)
#         self.decoder_relu2 = nn.ReLU(inplace=True)
#         self.decoder_res2 = ResidualBlock(nf)
#     def forward(self, x):
#         out1 = self.encoder_res1(x)
#
#         out2 = self.encoder_relu1(self.encoder_conv1(out1))
#         out3 = self.encoder_res2(out2)
#
#         out4 = self.encoder_relu2(self.encoder_conv2(out3))
#         out5 = self.encoder_res3(out4)
#
#         out6 = F.interpolate(out5, size=[out3.size()[2], out3.size()[3]], mode='bilinear')
#         out7 = self.decoder_relu1(self.decoder_conv1(out6))
#         out8 = self.decoder_res1(out7 + out3)
#
#         out9 = F.interpolate(out8, size=[out1.size()[2], out1.size()[3]], mode='bilinear')
#         out10 = self.decoder_relu2(self.decoder_conv2(out9))
#         out11 = self.decoder_res2(out10 + out1)
#
#         return out11
#
#
# class LightNet(nn.Module):
#     def __init__(self, nf):
#         super(LightNet, self).__init__()
#         self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.in2 = nn.InstanceNorm2d(nf, affine=True)
#         self.relu2 = nn.ReLU(inplace=True)
#
#         self.conv3 = nn.Conv2d(nf, nf * 2, 3, 2, 1, bias=True)
#         self.in3 = nn.InstanceNorm2d(nf * 2, affine=True)
#         self.relu3 = nn.ReLU(inplace=True)
#         self.conv4 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)
#         self.in4 = nn.InstanceNorm2d(nf * 2, affine=True)
#         self.relu4 = nn.ReLU(inplace=True)
#
#         self.conv5 = nn.Conv2d(nf * 2, nf * 4, 3, 2, 1, bias=True)
#         self.in5 = nn.InstanceNorm2d(nf * 4, affine=True)
#         self.relu5 = nn.ReLU(inplace=True)
#         self.conv6 = nn.Conv2d(nf * 4, nf * 4, 3, 1, 1, bias=True)
#         self.in6 = nn.InstanceNorm2d(nf * 4, affine=True)
#         self.relu6 = nn.ReLU(inplace=True)
#
#         self.conv7 = nn.Conv2d(nf * 4, nf * 2, 1, 1, 0, bias=True)
#         self.in7 = nn.InstanceNorm2d(nf * 2, affine=True)
#         self.relu7 = nn.ReLU(inplace=True)
#         self.conv8 = nn.Conv2d(nf * 2, nf * 2, 3, 1, 1, bias=True)
#         self.in8 = nn.InstanceNorm2d(nf * 2, affine=True)
#         self.relu8 = nn.ReLU(inplace=True)
#
#         self.conv9 = nn.Conv2d(nf * 2, nf, 1, 1, 0, bias=True)
#         self.in9 = nn.InstanceNorm2d(nf, affine=True)
#         self.relu9 = nn.ReLU(inplace=True)
#         self.conv10 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.in10 = nn.InstanceNorm2d(nf, affine=True)
#         self.relu10 = nn.ReLU(inplace=True)
#     def forward(self, x):
#         out2 = self.relu2(self.in2(self.conv2(x)))
#
#         out3 = self.relu3(self.in3(self.conv3(out2)))
#         out4 = self.relu4(self.in4(self.conv4(out3)))
#
#         out5 = self.relu5(self.in5(self.conv5(out4)))
#         out6 = self.relu6(self.in6(self.conv6(out5)))
#
#         up1 = F.interpolate(out6, size=[out4.size()[2], out4.size()[3]], mode='bilinear')
#         out7 = self.relu7(self.in7(self.conv7(up1)))
#         out8 = self.relu8(self.in8(self.conv8(out7 + out4)))
#
#         up2 = F.interpolate(out8, size=[out2.size()[2], out2.size()[3]], mode='bilinear')
#         out9 = self.relu9(self.in9(self.conv9(up2)))
#         out10 = self.relu10(self.in10(self.conv10(out9 + out2)))
#
#         return out10
#
#
# class LightenNet(nn.Module):
#     def __init__(self, in_channels, out_channels, nf, kernel_size=3, stride=1, padding=1, bias=False):
#         super(LightenNet, self).__init__()
#         # ----------------------------------------- Encoder ---------------------------------------- #
#         self.conv_in = nn.Conv2d(in_channels, nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#         self.relu_in = nn.ReLU()
#         self.Encoder_AINResBlock1 = AIN_ResBlock(nf)
#         self.Encoder_AINResBlock2 = AIN_ResBlock(nf)
#
#         self.Encoder_DownSample1 = DownSample(nf)
#         self.Encoder_AINResBlock3 = AIN_ResBlock(nf * 2)
#         self.Encoder_AINResBlock4 = AIN_ResBlock(nf * 2)
#         self.Encoder_DownSample2 = DownSample(nf * 2)
#
#         self.Encoder_AINSDResBlock1 = AIN_SDResBlock(nf * 4, 2)
#         self.Encoder_AINSDResBlock2 = AIN_SDResBlock(nf * 4, 2)
#         self.Encoder_AINSDResBlock3 = AIN_SDResBlock(nf * 4, 4)
#         self.Encoder_AINSDResBlock4 = AIN_SDResBlock(nf * 4, 4)
#         self.Encoder_AINSDResBlock5 = AIN_SDResBlock(nf * 4, 8)
#         self.Encoder_AINSDResBlock6 = AIN_SDResBlock(nf * 4, 8)
#         # ------------------------------------------------------------------------------------------ #
#
#         # ------------------------------------- Texture Branch ------------------------------------- #
#         self.Texture_input = Get_gradient_nopadding()
#         self.Texture_conv_in = nn.Conv2d(3, nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#         self.Texture_relu_in = nn.ReLU()
#         self.Texture_dam1 = DoubleAttentionModule(nf)
#         self.Texture_dam2 = DoubleAttentionModule(nf * 2)
#         self.Texture_dam3 = DoubleAttentionModule(nf * 4)
#         self.Texture_dam2_conv = nn.Conv2d(nf * 2, nf, kernel_size=1, stride=1, padding=0, bias=bias)
#         self.Texture_dam3_conv = nn.Conv2d(nf * 4, nf, kernel_size=1, stride=1, padding=0, bias=bias)
#         self.Texture_dam_add_conv = nn.Conv2d(nf, nf, kernel_size=1, stride=1, padding=0, bias=bias)
#         self.Texture_dam2_norm = nn.InstanceNorm2d(nf, affine=True)
#         self.Texture_dam3_norm = nn.InstanceNorm2d(nf, affine=True)
#         self.Texture_dam_add_norm = nn.InstanceNorm2d(nf, affine=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.Texture_net = TextureNet(nf)
#         self.Texture_conv_out = nn.Conv2d(nf, 3, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#         self.Texture_sigmoid = nn.Sigmoid()
#         # ------------------------------------------------------------------------------------------ #
#
#         # ------------------------------------- Light   Branch ------------------------------------- #
#         self.Light_conv_in = nn.Conv2d(1, nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#         self.Light_relu_in = nn.ReLU()
#         self.Light_net = LightNet(nf)
#         self.Light_conv_out = nn.Conv2d(nf, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#         self.Light_sigmoid = nn.Sigmoid()
#         # ------------------------------------------------------------------------------------------ #
#
#         # ----------------------------------------- Decoder ---------------------------------------- #
#         self.Decoder_conv1x1_1 = Conv1x1(nf * 4, nf * 2)
#
#         self.Decoder_texture_downsample = DownSample(nf)
#         self.Decoder_tadd_conv1 = nn.Conv2d(nf * 2, nf * 2, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.Decoder_tadd_norm1 = nn.InstanceNorm2d(nf * 2, affine=True)
#         self.Decoder_maxpool1 = nn.MaxPool2d((2, 2), stride=(2, 2))
#         self.Decoder_maxpool2 = nn.MaxPool2d((2, 2), stride=(2, 2))
#         self.Decoder_ainresblock1 = AIN_ResBlock(nf * 2)
#         self.Decoder_ainresblock2 = AIN_ResBlock(nf * 2)
#
#         self.Decoder_conv1x1_2 = Conv1x1(nf * 2, nf)
#
#         self.Decoder_tadd_conv2 = nn.Conv2d(nf, nf, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#         self.Decoder_tadd_norm2 = nn.InstanceNorm2d(nf, affine=True)
#         self.Decoder_ainresblock3 = AIN_ResBlock(nf)
#         self.Decoder_ainresblock4 = AIN_ResBlock(nf)
#         self.Decoder_conv_out = nn.Conv2d(nf, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
#         self.Decoder_sigmoid = nn.Sigmoid()
#         # ------------------------------------------------------------------------------------------ #
#
#     def forward(self, x):
#         # light
#         l1 = torch.clamp(torch.max(x, dim=1)[0].unsqueeze(1), min=0.0, max=1.0)
#         l2 = self.Light_relu_in(self.Light_conv_in(l1))
#         l3 = self.Light_net(l2)
#         l4 = self.Light_conv_out(l3)
#         lout = self.Light_sigmoid(l4 + l1)
#
#         # encoder
#         e1 = self.relu_in(self.conv_in(x))
#         e2 = self.Encoder_AINResBlock1(e1, l4)
#         e3 = self.Encoder_AINResBlock2(e2, l4)
#         e4 = self.Encoder_DownSample1(e3)
#         e5 = self.Encoder_AINResBlock3(e4, F.interpolate(l4, size=[e4.size()[2], e4.size()[3]], mode='bilinear', align_corners=True))
#         e6 = self.Encoder_AINResBlock4(e5, F.interpolate(l4, size=[e4.size()[2], e4.size()[3]], mode='bilinear', align_corners=True))
#         e7 = self.Encoder_DownSample2(e6)
#         e8 = self.Encoder_AINSDResBlock1(e7, l4)
#         e9 = self.Encoder_AINSDResBlock2(e8, l4)
#         e10 = self.Encoder_AINSDResBlock3(e9, l4)
#         e11 = self.Encoder_AINSDResBlock4(e10, l4)
#         e12 = self.Encoder_AINSDResBlock5(e11, l4)
#         e13 = self.Encoder_AINSDResBlock6(e12, l4)
#
#         # gradient
#         t1 = torch.clamp(self.Texture_input(x), min=0.0, max=1.0)
#         t2 = self.Texture_relu_in(self.Texture_conv_in(t1))
#         t3 = self.Texture_dam1(e1)
#         t4 = self.Texture_dam2(e4)
#         t5 = self.Texture_dam3(e7)
#         ta1 = F.interpolate(t4, size=[t3.size()[2], t3.size()[3]], mode='bilinear', align_corners=True)
#         ta2 = F.interpolate(t5, size=[t3.size()[2], t3.size()[3]], mode='bilinear', align_corners=True)
#         toa1 = self.relu(self.Texture_dam2_norm(self.Texture_dam2_conv(ta1)))
#         toa2 = self.relu(self.Texture_dam3_norm(self.Texture_dam3_conv(ta2)))
#         t6 = self.relu(self.Texture_dam_add_norm(self.Texture_dam_add_conv(t3 + toa1 + toa2)))
#         t7 = self.Texture_net(t2 + t6)
#         t8 = self.Texture_conv_out(t7)
#         tout = self.Texture_sigmoid(t8)
#
#         # decoder
#         d1 = self.Decoder_conv1x1_1(F.interpolate(e13, size=[e4.size()[2], e4.size()[3]], mode='bilinear', align_corners=True))
#         d2 = self.Decoder_texture_downsample(t7)
#         d3 = self.relu(self.Decoder_tadd_norm1(self.Decoder_tadd_conv1(d2 + d1)))
#         d4 = self.Decoder_ainresblock1(d3, F.interpolate(l4, size=[e4.size()[2], e4.size()[3]], mode='bilinear', align_corners=True))
#         d5 = self.Decoder_ainresblock2(d4, F.interpolate(l4, size=[e4.size()[2], e4.size()[3]], mode='bilinear', align_corners=True))
#         d6 = self.Decoder_conv1x1_2(F.interpolate(d5, size=[e3.size()[2], e3.size()[3]], mode='bilinear', align_corners=True))
#         d7 = self.relu(self.Decoder_tadd_norm2(self.Decoder_tadd_conv2(d6 + t7)))
#         d8 = self.Decoder_ainresblock3(d7, l4)
#         d9 = self.Decoder_ainresblock4(d8, l4)
#         d10 = self.Decoder_conv_out(d9)
#         dout = self.Decoder_sigmoid(d10 + x)
#
#         return t1, tout, l1, lout, dout
#
#
#     def initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.xavier_normal_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.InstanceNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight.data, 0, 1)
#                 m.bias.data.zero_()
#
# #
# # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #
# # net = LightenNet(3, 3, 32)
# # net.to(device)
# #
# # num_params = 0
# # for param in net.parameters():
# #     num_params += param.numel()
# # # print(net)
# # print('Total number of parameters: %d' % num_params)
# #
# # total = sum([param.nelement() for param in net.parameters()])
# # print(total / 1e6)
# # from torchsummary import summary
# # summary(net, input_size=(3, 251, 251))
#
#
#
