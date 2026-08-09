import random

import numpy as np
import skimage.color as sc

import torch

"""def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size
        ip = tp // scale
    else:
        tp = patch_size
        ip = patch_size

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    ret = [
        args[0][iy:iy + ip, ix:ix + ip, :],
        *[a[ty:ty + tp, tx:tx + tp, :] for a in args[1:]]
    ]

    return ret"""


def get_patch(*args, patch_size=96, scale=2, multi=False, input_large=False):
    ih, iw = args[0].shape[:2]

    if not input_large:
        p = scale if multi else 1
        tp = p * patch_size          # حجم HR patch المطلوب
        ip = tp // scale             # حجم LR patch المطلوب
    else:
        tp = patch_size
        ip = patch_size

    # اختيار عشوائي للإحداثيات مع التأكد من وجود مساحة كافية في LR
    # ولكن قد تكون حدود HR غير كافية بسبب اختلاف الحجم الأصلي للصورة
    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)

    if not input_large:
        tx, ty = scale * ix, scale * iy
    else:
        tx, ty = ix, iy

    # استخراج الـ LR patch (المفترض أن يكون بالحجم ip x ip)
    lr_patch = args[0][iy:iy + ip, ix:ix + ip, :]
    
    # استخراج الـ HR patch (قد يكون أصغر من tp x tp إذا تجاوز حدود الصورة)
    hr_patch = args[1][ty:ty + tp, tx:tx + tp, :] if len(args) > 1 else None

    # ========== التعديل الجديد (Padding) ==========
    # 1. التأكد من أن LR patch له الحجم ip x ip (يحدث نادراً)
    if lr_patch.shape[0] < ip or lr_patch.shape[1] < ip:
        pad_h = ip - lr_patch.shape[0]
        pad_w = ip - lr_patch.shape[1]
        lr_patch = np.pad(lr_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    
    # 2. التأكد من أن HR patch له الحجم tp x tp
    if hr_patch is not None and (hr_patch.shape[0] < tp or hr_patch.shape[1] < tp):
        pad_h = tp - hr_patch.shape[0]
        pad_w = tp - hr_patch.shape[1]
        hr_patch = np.pad(hr_patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    # =============================================

    ret = [lr_patch] + ([hr_patch] if hr_patch is not None else [])

    return ret

def set_channel(*args, n_channels=3):
    def _set_channel(img):
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)

        c = img.shape[2]
        if n_channels == 1 and c == 3:
            img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)
        elif n_channels == 3 and c == 1:
            img = np.concatenate([img] * n_channels, 2)

        return img

    return [_set_channel(a) for a in args]

def np2Tensor(*args, rgb_range=255):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        tensor = torch.from_numpy(np_transpose).float()
        tensor.mul_(rgb_range / 255)

        return tensor

    return [_np2Tensor(a) for a in args]

def augment(*args, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip: img = img[:, ::-1, :]
        if vflip: img = img[::-1, :, :]
        if rot90: img = img.transpose(1, 0, 2)
        
        return img

    return [_augment(a) for a in args]

