import torch
import torch.nn as nn
from ops.dcn.deform_conv import ModulatedDeformConv
import skimage.color as color
import skimage.segmentation as seg
import torch.nn.functional as F
import cv2
import numpy as np
from torch.nn import init
class FrameQE(nn.Module):
    def __init__(self):
        super(FrameQE, self).__init__()
        self.name = 'FrameQE'
        # Feature Extraction
        self.unet = unet(in_nc=1, nf=32, nb=3)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=True), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32*3, 32, 3, padding=1, bias=True), nn.ReLU())
        #CFBlocks
        self.cf = CF_Blocks()
        # Differential
        self.dif = Differential()
        # Reconstruction
        self.recon = Reconstruction()

    def forward(self, x):

        PFrm = x[:, 1, ...].unsqueeze(1)
        # Feature Extraction
        batch, num, h, w = x.shape
        y = x.view(-1, h, w).unsqueeze(1)
        u_out = self.unet(y)
        fea_P = self.conv1(PFrm)
        # CFBlocks
        out = u_out.view(batch, num, -1, h, w)
        out = self.cf(out)
        # Differential
        out2 = u_out.view(batch, -1, h, w)
        out2 = self.conv2(out2)
        in_diff = fea_P - out2
        dif_out = self.dif(in_diff)
        dif_out = dif_out + out2

        a = torch.cat([dif_out, out], dim=1)
        out = torch.add(self.recon(a), PFrm)
        return out

class Res_block(nn.Module):
    def __init__(self):
        super(Res_block, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
        self.cam = Cu_block2()
    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = res + x
        res = self.cam(res)
        return res

class Differential(nn.Module):
    def __init__(self):
        super(Differential, self).__init__()
        # Differential
        self.dif_conv1 = nn.Sequential(nn.Conv2d(32, 32, 5, padding=2, bias=True), nn.ReLU())
        self.dif_conv2 = nn.Sequential(nn.Conv2d(32, 32, 5, padding=2, stride=2, bias=True), nn.ReLU())
        self.dif_conv3 = nn.Sequential(nn.Conv2d(32, 32, 1, padding=0, bias=True), nn.ReLU())
        # self.cu = self.cu_layer(Cu_block, 3)
        self.cu1 = Cu_block()
        self.cu2 = Cu_block()
        self.cu3 = Cu_block()
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.dif_conv5 = nn.Sequential(nn.Conv2d(40, 32, 1, padding=0, bias=True), nn.ReLU())
    def forward(self, x):
        dif_out = self.dif_conv1(x)
        a = dif_out
        dif_out = self.dif_conv2(dif_out)
        dif_out = self.dif_conv3(dif_out)
        # dif_out = self.cu(dif_out)
        dif_out = self.cu1(dif_out)
        dif_out = self.cu2(dif_out)
        dif_out = self.cu3(dif_out)
        dif_out = self.pixel_shuffle(dif_out)
        dif_out = torch.cat([a, dif_out], 1)
        dif_out = self.dif_conv5(dif_out)
        return dif_out
    def cu_layer(self, cu_block, layer_nums):
        layer = []
        for _ in range(layer_nums):
            layer.append(cu_block())
        return nn.Sequential(*layer)
    def dw_mutil_scale(self, in_channel, out_channel, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=True, groups=in_channel),
            nn.ReLU()
        )

class Cu_block(nn.Module):
    def __init__(self):
        super(Cu_block, self).__init__()
        self.dw_conv_3 = self.dw_mutil_scale(32, 32, 3, 1)
        self.dw_conv_5 = self.dw_mutil_scale(32, 32, 5, 1)
        self.dw_conv_7 = self.dw_mutil_scale(32, 32, 7, 1)
        self.Fc_Sig = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Conv2d(32, 32, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 1),
            nn.Sigmoid()
        )
        self.dif_conv4 = nn.Sequential(nn.Conv2d(32, 32, 1, padding=0, bias=True), nn.ReLU())

    def forward(self, x):

        dif_out_3 = self.dw_conv_3(x)
        dif_out_5 = self.dw_conv_5(x)
        dif_out_7 = self.dw_conv_7(x)
        dif_out_357 = dif_out_3 + dif_out_5 + dif_out_7
        se_out = self.Fc_Sig(dif_out_357)
        se_out = dif_out_357 * se_out
        se_out = self.dif_conv4(se_out)
        dif_out = x + se_out
        return dif_out
    def dw_mutil_scale(self, in_channel, out_channel, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=True, groups=in_channel),
            nn.ReLU()
        )
class Reconstruction(nn.Module):
    def __init__(self):
        super(Reconstruction, self).__init__()
        self.name = 'Recon'

        # self.conv_3 = mutil_scale(64, 48, 3, 1, batchNorm=True)
        # self.conv_5 = mutil_scale(64, 48, 5, 1, batchNorm=True)
        # self.conv_7 = mutil_scale(64, 48, 7, 1, batchNorm=True)
        # self.recon_conv1 = conv_recon(False, 48 * 3, 48, kernel_size=3, stride=1)
        # self.recon_conv2 = conv_recon(False, 48, 48, kernel_size=3, stride=1)
        # self.recon_conv3 = conv_recon(False, 48 * 2, 48, kernel_size=3, stride=1)
        # self.recon_conv4 = conv_recon(False, 48 * 3, 48, kernel_size=3, stride=1)
        # self.recon_conv5 = conv_recon(False, 48 * 4, 48, kernel_size=3, stride=1)
        # self.recon_conv6 = conv_recon(False, 48, 1, kernel_size=3, stride=1)
        self.fea_extra = nn.Sequential(nn.Conv2d(32 * 2, 48, 3, padding=1, bias=True), nn.ReLU(inplace=True))
        self.recon_layer = res_layer(6)
        # self.recon_layer = self.res_layer(Res_block,6)
        self.conv = nn.Conv2d(48, 1, 3, stride=1, padding=1)

    def forward(self, y):
        # c_3 = self.conv_3(y)
        # c_5 = self.conv_5(y)
        # c_7 = self.conv_7(y)
        # y = torch.cat([c_3, c_5, c_7], dim=1)
        # cc1 = self.recon_conv1(y)
        # cc2 = self.recon_conv2(cc1)
        # ccc2 = torch.cat([cc1, cc2], dim=1)
        # cc3 = self.recon_conv3(ccc2)
        # ccc3 = torch.cat([cc1, cc2, cc3], dim=1)
        # cc4 = self.recon_conv4(ccc3)
        # ccc4 = torch.cat([cc1, cc2, cc3, cc4], dim=1)
        # cc5 = self.recon_conv5(ccc4)
        # cc6 = self.recon_conv6(cc5)
        # return cc6


        fea = self.fea_extra(y)
        fea = self.recon_layer(fea)
        out = self.conv(fea)
        return out

    def res_layer(self, res_block, layer_nums):
        res_layers = []
        for _ in range(layer_nums):
            res_layers.append(res_block())
        return nn.Sequential(*res_layers)

def res_layer(layer_nums):
    res_layers = []
    for _ in range(layer_nums):
        res_layers += [
                nn.Conv2d(48, 48, 3, padding=1),
                nn.ReLU(inplace=True)
                ]
    return nn.Sequential(*res_layers)
def conv_recon(batchNorm, in_channel, out_channel, kernel_size, stride):
    padding = (kernel_size - 1) // 2
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_channel, track_running_stats=True, affine=True, eps=0.001, momentum=0.99),
            nn.PReLU(out_channel)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.PReLU(out_channel)
        )
def mutil_scale(in_channel, out_channel, kernel_size, stride, batchNorm):
        padding = (kernel_size - 1) // 2
        if batchNorm:
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.BatchNorm2d(out_channel, track_running_stats=True, affine=True, eps=0.001, momentum=0.99),
                nn.PReLU(out_channel)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.PReLU(out_channel)
            )
class convBlock(nn.Module):
    def __init__(self):
        super(convBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(48, 48, 3, stride=1, padding=1),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class unet(nn.Module):
    def __init__(self, in_nc, nf, nb, base_ks=3):
        super(unet,self).__init__()
        self.nb = nb
        self.in_nc = in_nc
        # u-shape backbone
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
        for i in range(1, nb):
            setattr(
                self, 'dn_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True)
                )
            )
            setattr(
                self, 'up_conv{}'.format(i), nn.Sequential(
                    nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.ReLU(inplace=True)
        )
    def forward(self, inputs):
        nb = self.nb
        # feature extraction (with downsampling)
        out_lst = [self.in_conv(inputs)]  # record feature maps for skip connections
        for i in range(1, nb):
            dn_conv = getattr(self, 'dn_conv{}'.format(i))
            out_lst.append(dn_conv(out_lst[i - 1]))
        # trivial conv
        out = self.tr_conv(out_lst[-1])
        # feature reconstruction (with upsampling)
        for i in range(nb - 1, 0, -1):
            up_conv = getattr(self, 'up_conv{}'.format(i))
            out = up_conv(
                torch.cat([out, out_lst[i]], 1)
            )
        return out

class Cu_block2(nn.Module):
    def __init__(self):
        super(Cu_block2, self).__init__()
        self.dw_conv_3 = self.dw_mutil_scale(64, 64, 3, 1)
        self.dw_conv_5 = self.dw_mutil_scale(64, 64, 5, 1)
        self.dw_conv_7 = self.dw_mutil_scale(64, 64, 7, 1)
        self.Fc_Sig = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling
            nn.Conv2d(64, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 1),
            nn.Sigmoid()
        )
        self.dif_conv4 = nn.Sequential(nn.Conv2d(64, 64, 1, padding=0, bias=True), nn.ReLU())

    def forward(self, x):

        dif_out_3 = self.dw_conv_3(x)
        dif_out_5 = self.dw_conv_5(x)
        dif_out_7 = self.dw_conv_7(x)
        dif_out_357 = dif_out_3 + dif_out_5 + dif_out_7
        se_out = self.Fc_Sig(dif_out_357)
        se_out = dif_out_357 * se_out
        se_out = self.dif_conv4(se_out)
        dif_out = x + se_out
        return dif_out
    def dw_mutil_scale(self, in_channel, out_channel, kernel_size, stride):
        padding = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                      stride=stride, padding=padding, bias=True, groups=in_channel),
            nn.ReLU()
        )


    batch, num, h, w = x.shape
    x = x.view(-1, h, w)
    out_g = []
    for i in range(batch*num):
        a = x[i].cpu().numpy()*255
        a = a.astype(np.uint8)
        blur = cv2.GaussianBlur(a, (11, 11), 0)
        gaussImg = a - blur
        # gaussImg = cv2.Canny(blur, 10, 70)
        # cv2.imwrite('blur.jpg', blur)
        # cv2.imwrite('edge.jpg', gaussImg)
        out_g.append(gaussImg/255.0)
    out_g = torch.from_numpy(np.array(out_g).astype(np.float32)).cuda()
    out_g = out_g.view(batch, num, h, w);

    return out_g

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv_d1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_d2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.conv_s1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_s2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        initialize_weights([self.conv_d1, self.conv_d2, self.conv_s1, self.conv_s2], 0.1)

    def forward(self, front, center, back):
        identity_f = front
        out_f = F.relu(self.conv_d1(front), inplace=True)
        out_f = self.conv_d2(out_f)

        identity_b = back
        out_b = F.relu(self.conv_d1(back), inplace=True)
        out_b = self.conv_d2(out_b)

        identity_c = center
        out_c = F.relu(self.conv_s1(center), inplace=True)
        out_c = self.conv_s2(out_c)

        fc = out_f + out_c
        bc = out_b + out_c
        return fc + identity_f, out_f + out_c + out_b + identity_c, bc + identity_b

def initialize_weights(net_l, scale=0.1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class CF_Blocks(nn.Module):
    def __init__(self):
        super(CF_Blocks, self).__init__()
        self.CF1 = ResidualBlock_noBN(32)
        self.CF2 = ResidualBlock_noBN(32)
        self.CF3 = ResidualBlock_noBN(32)
        self.CF4 = ResidualBlock_noBN(32)
        self.CF5 = ResidualBlock_noBN(32)
        self.CF6 = ResidualBlock_noBN(32)
        self.CF7 = ResidualBlock_noBN(32)
        self.CF8 = ResidualBlock_noBN(32)
        self.CF9 = ResidualBlock_noBN(32)
        self.recon1 = nn.Conv2d(32 * 3, 32, 3, 1, padding=1)
        self.recon2 = nn.Conv2d(32, 32, 3, 1, padding=1)
    def forward(self, y):

        front = y[:, 0, ...]
        back = y[:, 2, ...]
        center = y[:, 1, ...]

        out_f, out_c, out_b = self.CF1(front, center, back)
        out_f, out_c, out_b = self.CF2(out_f, out_c, out_b)
        out_f, out_c, out_b = self.CF3(out_f, out_c, out_b)
        out_f, out_c, out_b = self.CF4(out_f, out_c, out_b)
        out_f, out_c, out_b = self.CF5(out_f, out_c, out_b)
        out_f, out_c, out_b = self.CF6(out_f, out_c, out_b)
        out_f, out_c, out_b = self.CF7(out_f, out_c, out_b)
        out_f, out_c, out_b = self.CF8(out_f, out_c, out_b)
        out_f, out_c, out_b = self.CF9(out_f, out_c, out_b)
        a = torch.cat((out_f, out_c, out_b), 1)

        #offset = a.detach().cpu().numpy()
        #np.save('Myoffset.npy', offset)

        out = self.recon1(a)

        out = self.recon2(out)
        return out


