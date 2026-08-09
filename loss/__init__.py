import os
from importlib import import_module

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================
# FFT Loss — يُعاقب على فقدان الترددات العالية
# ======================================================
class FFTLoss(nn.Module):
    """
    Frequency-domain L1 loss على الـ magnitude spectrum.

    لماذا magnitude فقط؟
    - magnitude = قوة كل تردد (الحواف، الـ texture، التفاصيل الدقيقة)
    - phase = موضع هذه الترددات، حساس جداً ويُصعّب التدريب

    لماذا rfft2 وليس fft2؟
    - rfft2 يستغل تماثل الصور الحقيقية → أسرع بمرتين + ذاكرة أقل
    - النتيجة مطابقة لـ fft2[:, :, :W//2+1]
    """
    def __init__(self):
        super(FFTLoss, self).__init__()

    def forward(self, sr, hr):
        # تحويل فورييه للصورتين
        sr_fft = torch.fft.rfft2(sr,  norm='ortho')
        hr_fft = torch.fft.rfft2(hr,  norm='ortho')

        # الفرق في الـ magnitude (قوة الترددات) فقط
        sr_mag = torch.abs(sr_fft)
        hr_mag = torch.abs(hr_fft)

        return F.l1_loss(sr_mag, hr_mag)


# ======================================================
# Huber Loss — تجمع بين L1 و L2 (أقل حساسية للشواذ)
# ======================================================
class HuberLoss(nn.Module):
    """
    Huber loss مُصحَّحة لـ rgb_range=1
    
    المشكلة القديمة: delta=1.0 مع rgb_range=1
    → pixel values في [0,1] → كل الأخطاء < 1
    → Huber يُصبح L2 بالكامل ويفقد ميزة L1
    
    الإصلاح: delta=0.01 مع rgb_range=1
    → L2 للأخطاء الصغيرة جداً (< 0.01) = ~2.5% من pixels
    → L1 للأخطاء الأكبر (0.01~1.0) = باقي الـ pixels
    → توازن حقيقي بين L1 و L2
    
    الاستخدام: --loss "1*Huber"
    """
    def __init__(self, delta=0.01):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, sr, hr):
        return F.huber_loss(sr, hr, reduction='mean', delta=self.delta)


# ======================================================
# Charbonnier Loss — نسخة سلسة من L1
# ======================================================
class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (pseudo-Huber): sqrt( (x-y)^2 + eps^2 )
    تعطي نتائج أفضل من L1 في مهام SR وتكون أقل حساسية للضوء.
    """
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, sr, hr):
        diff = sr - hr
        loss = torch.sqrt(diff ** 2 + self.eps ** 2)
        return loss.mean()


class Loss(nn.modules.loss._Loss):
    def __init__(self, args, ckp):
        super(Loss, self).__init__()
        print('Preparing loss function:')

        self.n_GPUs = args.n_GPUs
        self.loss = []
        self.loss_module = nn.ModuleList()
        for loss in args.loss.split('+'):
            weight, loss_type = loss.split('*')
            if loss_type == 'MSE' or loss_type == 'L2':
                loss_function = nn.MSELoss()
            elif loss_type == 'L1':
                loss_function = nn.L1Loss()
            elif loss_type == 'FFT':
                # Frequency-Aware Loss — يُعاقب على فقدان الترددات العالية
                # الاستخدام: --loss "1*L1+0.05*FFT"
                # الوزن الموصى به: 0.03 ~ 0.10
                loss_function = FFTLoss()
            elif loss_type == 'Huber':
                # Huber loss: دلتا = 1.0 (يمكن تعديلها لاحقاً)
                loss_function = HuberLoss(delta=1.0)
            elif loss_type == 'Charbonnier':
                # Charbonnier loss (نسخة سلسة من L1)
                loss_function = CharbonnierLoss(eps=1e-3)
            elif loss_type.find('VGG') >= 0:
                module = import_module('loss.vgg')
                loss_function = getattr(module, 'VGG')(
                    loss_type[3:],
                    rgb_range=args.rgb_range
                )
            elif loss_type.find('GAN') >= 0:
                module = import_module('loss.adversarial')
                loss_function = getattr(module, 'Adversarial')(
                    args,
                    loss_type
                )

            self.loss.append({
                'type': loss_type,
                'weight': float(weight),
                'function': loss_function}
            )
            if loss_type.find('GAN') >= 0:
                self.loss.append({'type': 'DIS', 'weight': 1, 'function': None})

        if len(self.loss) > 1:
            self.loss.append({'type': 'Total', 'weight': 0, 'function': None})

        for l in self.loss:
            if l['function'] is not None:
                print('{:.3f} * {}'.format(l['weight'], l['type']))
                self.loss_module.append(l['function'])

        self.log = torch.Tensor()

        device = torch.device('cpu' if args.cpu else 'cuda')
        self.loss_module.to(device)
        if args.precision == 'half': self.loss_module.half()
        if not args.cpu and args.n_GPUs > 1:
            self.loss_module = nn.DataParallel(
                self.loss_module, range(args.n_GPUs)
            )

        if args.load != '': self.load(ckp.dir, cpu=args.cpu)

    def forward(self, sr, hr):
        losses = []
        for i, l in enumerate(self.loss):
            if l['function'] is not None:
                loss = l['function'](sr, hr)
                effective_loss = l['weight'] * loss
                losses.append(effective_loss)
                self.log[-1, i] += effective_loss.item()
            elif l['type'] == 'DIS':
                self.log[-1, i] += self.loss[i - 1]['function'].loss

        loss_sum = sum(losses)
        if len(self.loss) > 1:
            self.log[-1, -1] += loss_sum.item()

        return loss_sum

    def step(self):
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                l.scheduler.step()

    """def start_log(self):
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))"""
    
    def start_log(self):
    # التحقق من أن self.log ليس فارغاً أو له شكل خاطئ
        if not hasattr(self, 'log') or self.log.numel() == 0:
            # تهيئة السجل بعدد الأعمدة المناسب (عدد مكونات الخسارة)
            self.log = torch.zeros(0, len(self.loss))
        
        # إذا كان عدد الأعمدة الحالي لا يساوي len(self.loss) ، أعد التهيئة
        if self.log.shape[1] != len(self.loss):
            self.log = torch.zeros(self.log.shape[0], len(self.loss))
        
        # إضافة صف جديد من الأصفار بعدد الأعمدة الصحيح
        self.log = torch.cat((self.log, torch.zeros(1, len(self.loss))))

    def end_log(self, n_batches):
        self.log[-1].div_(n_batches)

    def display_loss(self, batch):
        n_samples = batch + 1
        log = []
        for l, c in zip(self.loss, self.log[-1]):
            log.append('[{}: {:.4f}]'.format(l['type'], c / n_samples))
        return ''.join(log)

    def plot_loss(self, apath):
        # ✅ إصلاح: المحور من len(self.log) بدل epoch parameter
        # - رسم متواصل بعد كل resume بدون انقطاع
        # - لا يحتاج epoch كـ argument من utility.py
        total_epochs = len(self.log)
        if total_epochs == 0:
            return
        axis = np.linspace(1, total_epochs, total_epochs)

        for i, l in enumerate(self.loss):
            label = '{} Loss'.format(l['type'])
            fig = plt.figure()
            plt.title(label)
            plt.plot(axis, self.log[:, i].numpy(), label=label)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.grid(True)
            try:
                plt.savefig(os.path.join(apath, 'loss_{}.pdf'.format(l['type'])))
            except PermissionError:
                alt = os.path.join(apath, 'loss_{}_alt.pdf'.format(l['type']))
                plt.savefig(alt)
            plt.close(fig)

    def get_loss_module(self):
        if self.n_GPUs == 1:
            return self.loss_module
        else:
            return self.loss_module.module

    def save(self, apath):
        torch.save(self.state_dict(), os.path.join(apath, 'loss.pt'))
        torch.save(self.log, os.path.join(apath, 'loss_log.pt'))

    def load(self, apath, cpu=False):
        if cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        self.load_state_dict(torch.load(
            os.path.join(apath, 'loss.pt'),
            **kwargs
        ))
        self.log = torch.load(os.path.join(apath, 'loss_log.pt'))
        for l in self.get_loss_module():
            if hasattr(l, 'scheduler'):
                for _ in range(len(self.log)): l.scheduler.step()