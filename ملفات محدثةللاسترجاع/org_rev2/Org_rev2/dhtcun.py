import torch
import torch.nn as nn
from . import dhtcu_block as B
def make_model(args, parent=False):
    #model = HUTCN()
    #model = HUTCN(upscale = args.scale[0])
    model = HUTCN(upscale=args.scale[0],nf=args.n_feats)
    return model


"""class Cascade(nn.Module):
    def __init__(self, ):
        super(Cascade, self).__init__()
        self.conv1 = B.conv_layer(50, 50, kernel_size=1)
        self.conv3 = B.conv_layer(50, 50, kernel_size=3)
        self.conv5 = B.conv_layer(50, 50, kernel_size=5)
        self.c = B.conv_block(50 * 4, 50, kernel_size=1, act_type='lrelu')

    def forward(self, x):
        conv5 = self.conv5(x)
        extra = x+conv5
        conv3 = self.conv3(extra)
        extra = x + conv3
        conv1 = self.conv1(extra)
        cat = torch.cat([conv5, conv3, conv1, x], dim=1)
        input = self.c(cat)
        return input"""


class HUTCN(nn.Module):
    def __init__(self, in_nc=3, nf=50, num_modules=4, out_nc=3, upscale=3):
        super(HUTCN, self).__init__()

        self.fea_conv = B.conv_layer(in_nc, nf, kernel_size=1)
        # [تصحيح #1+2]: بعد U-Net تقول الورقة: ESA ثم Conv1x1
        # "features pass through Enhanced spatial attention block followed by 1x1 conv"
        self.post_unet_esa  = B.ESA(nf, nn.Conv2d)
        self.post_unet_conv = B.conv_layer(nf, nf, kernel_size=1)

        self.B1 = B.P_HTCB(in_channels=nf)
        self.B2 = B.P_HTCB(in_channels=nf)
        self.B3 = B.P_HTCB(in_channels=nf)
        self.B4 = B.P_HTCB(in_channels=nf)
        self.B5 = B.P_HTCB(in_channels=nf)
        #self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')
        #self.LR_conv = B.conv_layer(nf, nf, kernel_size=1)
        self.LR_conv1 = B.conv_layer(nf, nf, kernel_size=1)
        self.LR_conv2 = B.conv_layer(nf, nf, kernel_size=1)
        # [تصحيح #3]: الورقة تقول PixelShuffle ثم طبقتان Conv3x3
        # "Pixel Shuffle followed by two layers of the 3×3 convolution"
        # pixelshuffle_block الحالي يعمل: Conv3x3 → PixelShuffle (خاطئ)
        # التصحيح: PixelShuffle → Conv3x3 → Conv3x3
        self.pixel_shuffle = nn.PixelShuffle(upscale)
        #self.ps_conv1 = B.conv_layer(out_nc, out_nc, kernel_size=3)
        #self.ps_conv2 = B.conv_layer(out_nc, out_nc, kernel_size=3)
        # طبقة لرفع channels قبل PixelShuffle
        #self.pre_shuffle = B.conv_layer(nf, out_nc * (upscale ** 2), kernel_size=3)
        
        #############################################################
        self.recon_conv1     = B.conv_layer(nf, nf, kernel_size=3)   # Conv3×3 أولى
        self.recon_conv2     = B.conv_layer(nf, out_nc * (upscale**2), kernel_size=3)  # Conv3×3 ثانية
        #####################################################

        self.scale_idx = 0

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)

        out_B4 = self.B4(out_B3)
        out_B4 = self.LR_conv1(out_B4) + out_B2

        out_B5 = self.B5(out_B4)
        out_B5 = self.LR_conv2(out_B5) + out_B1

        # [تصحيح #1+2]: الورقة تقول بعد U-Net: ESA ثم Conv1x1 ثم + residual
        # كان: out_lr = self.c1_r(out_B5) + out_fea  ← Conv3x3 خاطئ
        out_lr = self.post_unet_conv(self.post_unet_esa(out_B5)) + out_fea

        # [تصحيح #3]: PixelShuffle ثم Conv3x3 ثم Conv3x3
        # كان: self.upsampler(out_lr) ← Conv3x3→PixelShuffle خاطئ
        #out_ps = self.pixel_shuffle(self.pre_shuffle(out_lr))
        #output = self.ps_conv2(self.ps_conv1(out_ps))

    ###########################################################
        out_r1  = self.recon_conv1(out_lr)
        out_r2  = self.recon_conv2(out_r1)
        return self.pixel_shuffle(out_r2)
    #####################################################

        #return output
    
    
    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
