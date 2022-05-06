import torch
import torch.nn as nn
import torch.nn.functional as F
from Code.model_lung_infection.backbone.Res2Net import res2net50_v1b_26w_4s


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
class RFB_modified(nn.Module):           #改进的RBF模块
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
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, n_class, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class Inf_Net(nn.Module):
    def __init__(self, channel=32, n_class=1):
        super(Inf_Net, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # 融合各层特征
        self.conv1down = nn.Conv2d(64, 16, 1, padding=0)
        self.conv2down = nn.Conv2d(256, 16, 1, padding=0)
        self.conv3down = nn.Conv2d(512, 16, 1, padding=0)
        self.conv4down = nn.Conv2d(1024, 16, 1, padding=0)
        self.conv5down = nn.Conv2d(2048, 16, 1, padding=0)

        self.score_dsn1 = nn.Conv2d(16, 1, 1)
        self.score_dsn2 = nn.Conv2d(16, 1, 1)
        self.score_dsn3 = nn.Conv2d(16, 1, 1)
        self.score_dsn4 = nn.Conv2d(16, 1, 1)
        self.score_dsn5 = nn.Conv2d(16, 1, 1)
        self.score_final = nn.Conv2d(5, 1, 1)
        self.edge_conv = BasicConv2d(5, 64, kernel_size=1)

        # ---- Receptive Field Block like module ----
        self.rfb3 = RFB_modified(512, channel)
        self.rfb4 = RFB_modified(1024, channel)
        self.rfb5 = RFB_modified(2048, channel)

        # ---- Partial Decoder ----
        self.ParDec = aggregation(channel, n_class)

        # ---- reverse attention branch 5 ----
        self.ra_h1_conv1 = BasicConv2d(2048, 512, kernel_size=1)
        self.ra_h1_conv2 = BasicConv2d(512 + 64, 256, kernel_size=5, padding=2)
        self.ra_h1_conv3 = BasicConv2d(256, n_class, kernel_size=1)
        # ---- reverse attention branch 4 ----
        self.ra_h2_conv1 = BasicConv2d(1024, 256, kernel_size=1)
        self.ra_h2_conv2 = BasicConv2d(256 + 64, 64, kernel_size=3, padding=1)
        self.ra_h2_conv3 = BasicConv2d(64, n_class, kernel_size=3, padding=1)
        # ---- reverse attention branch 3 ----
        self.ra_m2_conv1 = BasicConv2d(512, 256, kernel_size=1)
        self.ra_m2_conv2 = BasicConv2d(256 + 64, 64, kernel_size=5, padding=2)
        self.ra_m2_conv3 = BasicConv2d(64, n_class, kernel_size=1)
        # ---- reverse attention branch 2 ----
        self.ra_l1_conv1 = BasicConv2d(256, 64, kernel_size=1)
        self.ra_l1_conv2 = BasicConv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.ra_l1_conv3 = BasicConv2d(64, n_class, kernel_size=1)
        # ---- reverse attention branch 1 ----
        self.ra_l2_conv1 = BasicConv2d(64, 32, kernel_size=1)
        self.ra_l2_conv2 = BasicConv2d(32 + 64, 32, kernel_size=3, padding=1)
        self.ra_l2_conv3 = BasicConv2d(32, n_class, kernel_size=3, padding=1)
        # 降维
        # self.convmap4 = BasicConv2d(512,64,kernel_size=1)
        self.convmap = BasicConv2d(5, n_class, kernel_size=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x1 = self.resnet.relu(x)            #bs,54,176,176

        # ---- low-level features ----
        x2 = self.resnet.maxpool(x1)      # bs, 64, 88, 88
        x2 = self.resnet.layer1(x2)      # bs, 256, 88, 88

        # ---- high-level features ----
        x3 = self.resnet.layer2(x2)     # bs, 512, 44, 44
        x4 = self.resnet.layer3(x3)     # bs, 1024, 22, 22
        x5 = self.resnet.layer4(x4)     # bs, 2048, 11, 11

        x3_rfb = self.rfb3(x3)        # channel -> 32
        x4_rfb = self.rfb4(x4)        # channel -> 32
        x5_rfb = self.rfb5(x5)        # channel -> 32

        # ---- edge guidance ----     边界指导
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

        s1 = F.interpolate(s1_out, scale_factor=2, mode='bilinear')
        s2 = F.interpolate(s2_out, scale_factor=4, mode='bilinear')
        s3 = F.interpolate(s3_out, scale_factor=8, mode='bilinear')
        s4 = F.interpolate(s4_out, scale_factor=16, mode='bilinear')
        s5 = F.interpolate(s5_out, scale_factor=32, mode='bilinear')

        fusecat = torch.cat((s1, s2, s3, s4, s5), dim=1)  # 各层融合concat
        fuse = self.score_final(fusecat)
        edge_guidance = self.edge_conv(fusecat)  # [bs,64,352,352]
        lateral_edge = fuse  # [bs,1,352,352]


        # ---- global guidance ----    全局指导
        ra5_feat = self.ParDec(x5_rfb, x4_rfb, x3_rfb)          # 1,44,44
        lateral_map_5 = F.interpolate(ra5_feat,
                                      scale_factor=8,
                                      mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # 高层聚合 par → f5
        crop_5 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear') #11
        x = -1*(torch.sigmoid(crop_5)) + 1  # reverse
        x = x.expand(-1, 2048, -1, -1).mul(x5)
        x = torch.cat((self.ra_h1_conv1(x), F.interpolate(edge_guidance, scale_factor=1/32, mode='bilinear')), dim=1)
        x = F.relu(self.ra_h1_conv2(x))
        ra4_feat = self.ra_h1_conv3(x)
        x = ra4_feat + crop_5   # element-wise addition
        lateral_map_4 = F.interpolate(x,scale_factor=32,mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        #高层聚合 f5 → f4
        crop_4 = F.interpolate(x, scale_factor=2, mode='bilinear')      #22
        x = -1*(torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x4)
        x = torch.cat((self.ra_h2_conv1(x), F.interpolate(edge_guidance, scale_factor=1/16, mode='bilinear')), dim=1)
        x = F.relu(self.ra_h2_conv2(x))
        ra3_feat = self.ra_h2_conv3(x)
        x = ra3_feat + crop_4
        lateral_map_3 = F.interpolate(x,scale_factor=16,mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        #中层聚合 f4 → f3
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')      #44
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x3)
        x = torch.cat((self.ra_m2_conv1(x), F.interpolate(edge_guidance, scale_factor=1/8, mode='bilinear')), dim=1)
        x = F.relu(self.ra_m2_conv2(x))
        ra2_feat = self.ra_m2_conv3(x)
        x = ra2_feat + crop_3
        lateral_map_2 = F.interpolate(x,scale_factor=8,mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # 底层聚合 par → f2
        lm_1 = F.interpolate(x, scale_factor=2, mode='bilinear')  # 88
        x_lm_1 = -1 * (torch.sigmoid(lm_1)) + 1
        x_lm_1 = x_lm_1.expand(-1, 256, -1, -1).mul(x2)
        x_lm_1 = torch.cat((self.ra_l1_conv1(x_lm_1), F.interpolate(edge_guidance, scale_factor=1 / 4, mode='bilinear')), dim=1)
        x_lm_1 = F.relu(self.ra_l1_conv2(x_lm_1))
        low_map_2 = self.ra_l1_conv3(x_lm_1)
        x_2 = low_map_2 + lm_1
        lateral_map_1 = F.interpolate(x_2, scale_factor=4,mode='bilinear')  # NOTES: Sup-4 (bs, 1, 88, 88) -> (bs, 1, 352, 352)

        # 底层特征 f2 → f1
        lm_2   = F.interpolate(x_2, scale_factor=2, mode='bilinear')  # 176
        x_lm_2 = -1 * (torch.sigmoid(lm_2)) + 1
        x_lm_2 = x_lm_2.expand(-1, 64, -1, -1).mul(x1)
        x_lm_2 = torch.cat((self.ra_l2_conv1(x_lm_2),F.interpolate(edge_guidance, scale_factor=1 / 2, mode='bilinear')), dim=1)
        x_lm_2 = F.relu(self.ra_l2_conv2(x_lm_2))
        low_map_1 = self.ra_l2_conv3(x_lm_2)
        x_1 = low_map_1 + lm_2
        lateral_map_0 = F.interpolate(x_1, scale_factor=2,mode='bilinear')  # NOTES: Sup-4 (bs, 1, 176, 176) -> (bs, 1, 352, 352)

        final_map = lateral_map_4 + lateral_map_3 + lateral_map_2 + lateral_map_1 + lateral_map_0
        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2, lateral_map_1, lateral_map_0,final_map, lateral_edge


if __name__ == '__main__':
    ras = PraNetPlusPlus().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)
    print(out[0].shape)
