import torch
import torch.nn as nn
from . import dhtcu_block as B

def make_model(args, parent=False):
    model = HUTCN(upscale=args.scale[0], nf=args.n_feats)
    return model

class HUTCN(nn.Module):
    def __init__(self, in_nc=3, nf=96, num_modules=4, out_nc=3, upscale=4):
        super(HUTCN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=1)

        # بعد U-Net: ESA ثم Conv1x1
        self.post_unet_esa  = B.ESA(nf, nn.Conv2d)
        self.post_unet_conv = B.conv_layer(nf, nf, kernel_size=1)

        # 4 كتل P_HTCB (B1..B4)
        self.B1 = B.P_HTCB(in_channels=nf)
        self.B2 = B.P_HTCB(in_channels=nf)
        self.B3 = B.P_HTCB(in_channels=nf)
        self.B4 = B.P_HTCB(in_channels=nf)

        # طبقات LRConv (LR_conv1 مستخدمة، LR_conv2 موجودة للتوافق)
        self.LR_conv1 = B.conv_layer(nf, nf, kernel_size=1)
        self.LR_conv2 = B.conv_layer(nf, nf, kernel_size=1)

        # طبقات إعادة البناء: Conv3x3 → Conv3x3 → PixelShuffle
        self.recon_conv1 = B.conv_layer(nf, nf, kernel_size=3)
        self.recon_conv2 = B.conv_layer(nf, out_nc * (upscale**2), kernel_size=3)
        self.pixel_shuffle = nn.PixelShuffle(upscale)

    def forward(self, input):
        out_fea = self.fea_conv(input)

        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)

        # Residual داخل السلسلة (مطابق للكود الأصلي)
        out_B4 = self.LR_conv1(out_B4) + out_B3

        # بعد U-Net: ESA + Conv1x1 ثم ربط مع out_fea
        out_lr = self.post_unet_conv(self.post_unet_esa(out_B4)) + out_fea

        # مرحلة إعادة البناء: Conv3x3 → Conv3x3 → PixelShuffle
        out_r1 = self.recon_conv1(out_lr)
        out_r2 = self.recon_conv2(out_r1)
        output = self.pixel_shuffle(out_r2)

        return output