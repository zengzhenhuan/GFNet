import torch
import torch.nn as nn
import torch.nn.functional as F
from Code.model_lung_infection.backbone.VGGNet import B2_VGG


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel, n_class):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4        = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5        = nn.Conv2d(3*channel, n_class, 1)

    def forward(self, x1, x2, x3):       #x1为f5,  22 ，  x2为f4,   44，  x3为f3,   88
        x1_1 = x1  #512 22
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2  #  44
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1)))  * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

class mynetVGG(nn.Module):
    def __init__(self, channel=32, n_class=1):
        super(mynetVGG, self).__init__()
        # ---- ResNet Backbone ----
        # self.resnet = res2net50_v1b_26w_4s(pre_trained=True)
        self.vgg = B2_VGG()
        #融合各层特征
        self.conv1down = nn.Conv2d(64, 16, 1, padding=0)
        self.conv2down = nn.Conv2d(128, 16, 1, padding=0)
        self.conv3down = nn.Conv2d(256, 16, 1, padding=0)
        self.conv4down = nn.Conv2d(512, 16, 1, padding=0)
        self.conv5down = nn.Conv2d(512, 16, 1, padding=0)

        self.score_dsn1 = nn.Conv2d(16, 1, 1)
        self.score_dsn2 = nn.Conv2d(16, 1, 1)
        self.score_dsn3 = nn.Conv2d(16, 1, 1)
        self.score_dsn4 = nn.Conv2d(16, 1, 1)
        self.score_dsn5 = nn.Conv2d(16, 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)
        self.edge_conv = BasicConv2d(5, 64, kernel_size=1)

        # ---- Receptive Field Block like module ----
        self.rfb3 = RFB_modified(256, channel)
        self.rfb4 = RFB_modified(512, channel)
        self.rfb5 = RFB_modified(512, channel)

        # ---- Partial Decoder ----
        self.ParDec = aggregation(channel, n_class)

        # ---- reverse attention branch 5 ----
        self.ra_h1_conv1 = BasicConv2d(512, 256, kernel_size=1)
        self.ra_h1_conv2 = BasicConv2d(256+64, 256, kernel_size=5, padding=2)
        self.ra_h1_conv3 = BasicConv2d(256, n_class, kernel_size=1)
        # ---- reverse attention branch 4 ----
        self.ra_h2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra_h2_conv2 = BasicConv2d(64+64, 64, kernel_size=3, padding=1)
        self.ra_h2_conv3 = BasicConv2d(64, n_class, kernel_size=3, padding=1)
        # ---- reverse attention branch 3 ----
        self.ra_m2_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra_m2_conv2 = BasicConv2d(64 + 64, 64, kernel_size=5, padding=2)
        self.ra_m2_conv3 = BasicConv2d(64, n_class, kernel_size=1)
        # ---- reverse attention branch 2 ----
        self.ra_l1_conv1 = BasicConv2d(128, 64, kernel_size=1)
        self.ra_l1_conv2 = BasicConv2d(64+64, 64, kernel_size=3, padding=1)
        self.ra_l1_conv3 = BasicConv2d(64, n_class, kernel_size=1)
        # ---- reverse attention branch 1 ----
        self.ra_l2_conv1 = BasicConv2d(64, 32, kernel_size=1)
        self.ra_l2_conv2 = BasicConv2d(32+64, 32, kernel_size=3, padding=1)
        self.ra_l2_conv3 = BasicConv2d(32, n_class, kernel_size=3, padding=1)
        #降维
        # self.convmap4 = BasicConv2d(512,64,kernel_size=1)
        self.convmap = BasicConv2d(5,n_class,kernel_size=1)
        self.conv0 = BasicConv2d(512,n_class,kernel_size=1)

    def forward(self, x):
        x1 = self.vgg.conv1(x)   # torch.Size([1, 64, 352, 352])
        x2 = self.vgg.conv2(x1)  # torch.Size([1, 128, 176, 176])
        x3 = self.vgg.conv3(x2) # torch.Size([1, 256, 88, 88])
        x4 = self.vgg.conv4_1(x3)   # torch.Size([1, 512, 44, 44])
        x5 = self.vgg.conv5_1(x4)   # torch.Size([1, 512, 22, 22])


        x3_rfb = self.rfb3(x3)
        x4_rfb = self.rfb4(x4)        # channel -> 32         f4
        x5_rfb = self.rfb5(x5)        # channel -> 32         f5

        #边界指导
        conv1down = self.conv1down(x1)
        conv2down = self.conv2down(x2)
        conv3down = self.conv3down(x3)
        conv4down = self.conv4down(x4)
        conv5down = self.conv5down(x5)

        s1_out = self.score_dsn1(conv1down)
        s2_out = self.score_dsn2(conv2down)
        s3_out = self.score_dsn3(conv3down)
        s4_out = self.score_dsn4(conv4down)
        s5_out = self.score_dsn5(conv5down)

        s1 = s1_out
        s2 = F.interpolate(s2_out, scale_factor=2, mode='bilinear')
        s3 = F.interpolate(s3_out, scale_factor=4, mode='bilinear')
        s4 = F.interpolate(s4_out, scale_factor=8, mode='bilinear')
        s5 = F.interpolate(s5_out, scale_factor=16, mode='bilinear')

        fusecat = torch.cat((s1, s2, s3, s4, s5), dim=1)  # 各层融合concat
        fuse = self.score_final(fusecat)
        edge_guidance = self.edge_conv(fusecat)  # [bs,64,352,352]
        lateral_edge = fuse  # [bs,1,352,352]

        #高层全局指导

        high_map = self.ParDec(x5_rfb, x4_rfb,x3_rfb)   #88
        highmap = F.interpolate(high_map,scale_factor=4,mode='bilinear')    # 损失函数中的高层指导 NOTES: Sup-1 (bs, 1, 88, 88) -> (bs, 1, 352, 352)

        #高层聚合 par → f5

        hm_1 = F.interpolate(high_map, scale_factor=0.25, mode='bilinear')   #22
        x_hm_1 = -1*(torch.sigmoid(hm_1)) + 1  # reverse    #反相RA
        x_hm_1 = x_hm_1.expand(-1, 512, -1, -1).mul(x5)
        x_hm_1 = torch.cat((self.ra_h1_conv1(x_hm_1), F.interpolate(edge_guidance, scale_factor=1 / 16, mode='bilinear')), dim=1)
        x_hm_1 = F.relu(self.ra_h1_conv2(x_hm_1))
        high_map_5 = self.ra_h1_conv3(x_hm_1)
        x_5 = high_map_5 + hm_1   # element-wise addition
        lateral_map_5 = F.interpolate(x_5, scale_factor=16,mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # f5 → f4

        hm_2 = F.interpolate(x_5, scale_factor=2, mode='bilinear')     #44
        x_hm_2 = -1*(torch.sigmoid(hm_2)) + 1
        x_hm_2 = x_hm_2.expand(-1, 512, -1, -1).mul(x4)
        x_hm_2 = torch.cat((self.ra_h2_conv1(x_hm_2), F.interpolate(edge_guidance, scale_factor=1 / 8, mode='bilinear')), dim=1)
        x_hm_2 = F.relu(self.ra_h2_conv2(x_hm_2))
        high_map_4 = self.ra_h2_conv3(x_hm_2)
        x_4 = high_map_4 + hm_2
        lateral_map_4 = F.interpolate(x_4,scale_factor=8,mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        #  f4 → f3
        mm_2 = F.interpolate(x_4, scale_factor=2, mode='bilinear')  # 88
        x_mm_2 = -1 * (torch.sigmoid(mm_2)) + 1
        x_mm_2 = x_mm_2.expand(-1, 256, -1, -1).mul(x3)
        x_mm_2 = torch.cat((self.ra_m2_conv1(x_mm_2), F.interpolate(edge_guidance, scale_factor=1 / 4, mode='bilinear')), dim=1)
        x_mm_2 = F.relu(self.ra_m2_conv2(x_mm_2))
        mid_map_3 = self.ra_m2_conv3(x_mm_2)
        x_3 = mid_map_3 + mm_2
        lateral_map_3 = F.interpolate(x_3, scale_factor=4,
                                      mode='bilinear')  # NOTES: Sup-3 (bs, 1, 88, 22) -> (bs, 1, 352, 352)
        # f3 → f2

        lm_1 = F.interpolate(x_3, scale_factor=2, mode='bilinear')   #176
        x_lm_1 = -1*(torch.sigmoid(lm_1)) + 1
        x_lm_1 = x_lm_1.expand(-1, 128, -1, -1).mul(x2)
        x_lm_1 = torch.cat((self.ra_l1_conv1(x_lm_1), F.interpolate(edge_guidance, scale_factor=1 / 2, mode='bilinear')), dim=1)
        x_lm_1 = F.relu(self.ra_l1_conv2(x_lm_1))
        low_map_2 = self.ra_l1_conv3(x_lm_1)
        x_2 = low_map_2 + lm_1
        lateral_map_2 = F.interpolate(x_2,scale_factor=2,mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # f2 → f1
        lm_2 = F.interpolate(x_2, scale_factor=2, mode='bilinear')   #352
        x_lm_2 = -1 * (torch.sigmoid(lm_2)) + 1
        x_lm_2 = x_lm_2.expand(-1, 64, -1, -1).mul(x1)
        x_lm_2 = torch.cat((self.ra_l2_conv1(x_lm_2), edge_guidance), dim=1)
        x_lm_2 = F.relu(self.ra_l2_conv2(x_lm_2))
        low_map_1 = self.ra_l2_conv3(x_lm_2)
        x_1 = low_map_1 + lm_2
        lateral_map_1 = x_1

        # 特征融合
        return lateral_map_5, lateral_map_4, lateral_map_3,lateral_map_2, lateral_map_1, \
               highmap,lateral_edge#



if __name__ == '__main__':
    ras = PraNetPlusPlus().cuda()
    input_tensor = torch.randn(1,3, 352, 352).cuda()

    out = ras(input_tensor)
    print(out[0].shape, out[1].shape, out[2].shape, out[3].shape, out[4].shape)