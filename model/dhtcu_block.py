import torch.nn as nn
from collections import OrderedDict  # ✅ أضف هذا السطر
import torch
import torch.nn.functional as F
from . import SGBlock,FNet,Spartial_Attention,SwinT
def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)
def conv_layer2(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    return nn.Sequential(#400epoch 32.726 28.623
        nn.Conv2d(in_channels, int(in_channels * 0.5), 1, stride, bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5 * 0.5), 1, 1, bias=True),
        nn.Conv2d(int(in_channels * 0.5 * 0.5), int(in_channels * 0.5), (1, 3), 1, (0, 1),
                           bias=True),
        nn.Conv2d(int(in_channels * 0.5), int(in_channels * 0.5), (3, 1), 1, (1, 0), bias=True),
        nn.Conv2d(int(in_channels * 0.5), out_channels, 1, 1, bias=True)
    )

def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

# ... (الإضافات السابقة تبقى كما هي)

class ESA(nn.Module):
    """Enhanced Spatial Attention – مطابق للبنية في log.txt"""
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)   # stride=2
        self.conv3 = conv(f, f, kernel_size=3, stride=1, padding=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv4 = conv(f, n_feats, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = self.conv1(x)                     # (B, f, H, W)
        c1 = self.conv2(c1_)                    # (B, f, H/2, W/2)   stride=2
        c3 = self.relu(self.conv3(c1))          # (B, f, H/2, W/2)
        c3 = F.interpolate(c3, size=x.shape[2:], mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)                   # المسار الموازي
        c4 = self.conv4(c3 + cf)                # (B, n_feats, H, W)
        m = self.sigmoid(c4)
        return x * m


class TCN(nn.Module):
    """Transformer CNN Block – الترتيب: Conv3x3 ثم SwinT"""
    def __init__(self, in_channels):
        super(TCN, self).__init__()
        self.conv3 = conv_layer(in_channels, in_channels, kernel_size=3)
        self.swinT = SwinT.SwinT(n_feats=in_channels)

    def forward(self, x):
        return self.swinT(self.conv3(x))


class TESA(nn.Module):
    """Triple Enhanced Spatial Attention – اثنان ESA (كما في log.txt)"""
    def __init__(self, in_channels):
        super(TESA, self).__init__()
        self.esa1 = ESA(in_channels, nn.Conv2d)
        self.esa2 = ESA(in_channels, nn.Conv2d)

    def forward(self, x):
        return self.esa2(self.esa1(x))


class P_HTCB(nn.Module):
    """
    Parallel Hybrid Transformer CNN Block – طبقًا للورقة وبنية log.txt:
    HTESA = TESA_in(Hi)
    HTCN  = TCN(HTESA)
    HConv = Conv1x1 + LeakyReLU (c)
    Output = TESA_out(HConv) + Hi
    """
    def __init__(self, in_channels):
        super(P_HTCB, self).__init__()
        self.tesa_in = TESA(in_channels)
        self.tcn1 = TCN(in_channels)
        self.c = conv_block(in_channels, in_channels, kernel_size=1, act_type='lrelu')
        self.tesa_out = TESA(in_channels)

    def forward(self, x):
        h_tesa = self.tesa_in(x)
        h_tcn = self.tcn1(h_tesa)
        h_conv = self.c(h_tcn)
        out = self.tesa_out(h_conv)
        return out + x
    
def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)