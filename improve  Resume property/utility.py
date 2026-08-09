import os
import math
import time
import datetime
from multiprocessing import Process, Queue

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from scipy.signal import convolve2d, windows


def bg_target(queue):
    while True:
        if not queue.empty():
            filename, tensor = queue.get()
            if filename is None:
                break
            imageio.imwrite(filename, tensor.numpy())


class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart:
            self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0
        return ret

    def reset(self):
        self.acc = 0


class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        # ✅ مسار ديناميكي — يعمل مع أي اسم مجلد
        _src_dir = os.path.dirname(os.path.abspath(__file__))
        _exp_dir = os.path.join(_src_dir, 'experiment')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join(_exp_dir, args.save)
        else:
            self.dir = os.path.join(_exp_dir, args.save)
            if os.path.exists(self.dir):
                self.log = torch.load(
                    self.get_path('psnr_log.pt'),
                    weights_only=False
                )
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt')) else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')
        self.n_processes = 8

        # ================================================================
        # ✅ تحميل PSNR log مع دعم --resume N
        # الحالات:
        #   --resume -1  → يكمل من آخر epoch محفوظ
        #   --resume N   → يكمل من epoch N بالضبط (يقطع log عند N-1)
        #   --resume 0   → تدريب جديد
        # ================================================================
        if args.load:
            psnr_log_path = self.get_path('psnr_log.pt')
            if os.path.exists(psnr_log_path):
                full_log = torch.load(psnr_log_path, weights_only=False)

                if args.resume == -1:
                    if len(self.log) == 0:
                        self.log = full_log
                    print(f'Resuming from last epoch — log has {len(self.log)} epochs.')
                elif args.resume > 0:
                    n = args.resume - 1
                    if len(full_log) >= n:
                        self.log = full_log[:n]
                    else:
                        self.log = full_log
                    print(f'Resuming from epoch {args.resume} — log has {len(self.log)} epochs.')
                    if len(self.log) > 0:
                        best_val = self.log[:, 0, 0].max().item()
                        best_ep  = self.log[:, 0, 0].argmax().item() + 1
                        print(f'Best PSNR so far: {best_val:.3f} dB @ epoch {best_ep}')

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir)
        self.plot_psnr()
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        self.log_file.write(log + '\n')
        if refresh:
            self.log_file.close()
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        self.log_file.close()

    def plot_psnr(self):
        total_epochs = len(self.log)
        if total_epochs == 0:
            return
        axis = np.linspace(1, total_epochs, total_epochs)

        for idx_data, d in enumerate(self.args.data_test):
            label = 'SR on {}'.format(d)
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                plt.plot(
                    axis,
                    self.log[:, idx_data, idx_scale].numpy(),
                    label='Scale {}'.format(scale)
                )
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            try:
                plt.savefig(self.get_path('test_{}.pdf'.format(d)))
            except PermissionError:
                alt_path = self.get_path('test_{}_alt.pdf'.format(d))
                plt.savefig(alt_path)
            plt.close(fig)

    def begin_background(self):
        self.queue = Queue()
        self.process = [
            Process(target=bg_target, args=(self.queue,))
            for _ in range(self.n_processes)
        ]
        for p in self.process:
            p.start()

    def end_background(self):
        for _ in range(self.n_processes):
            self.queue.put((None, None))
        while not self.queue.empty():
            time.sleep(1)
        for p in self.process:
            p.join()

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )
            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                self.queue.put(('{}{}.png'.format(filename, p), tensor_cpu))


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    h_min = min(sr.shape[-2], hr.shape[-2])
    w_min = min(sr.shape[-1], hr.shape[-1])
    sr = sr[..., :h_min, :w_min]
    hr = hr[..., :h_min, :w_min]

    if hr.nelement() == 1:
        return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    return -10 * math.log10(mse)


def calc_ssim(img1, img2, scale=2, rgb_range=255, dataset=None):
    h_min = min(img1.shape[-2], img2.shape[-2])
    w_min = min(img1.shape[-1], img2.shape[-1])
    img1  = img1[..., :h_min, :w_min]
    img2  = img2[..., :h_min, :w_min]

    if dataset and dataset.dataset.benchmark:
        border = math.ceil(scale)
    else:
        border = math.ceil(scale) + 6

    if rgb_range != 255:
        img1 = img1 * 255.0 / rgb_range
        img2 = img2 * 255.0 / rgb_range

    img1 = img1.data.squeeze().float().clamp(0, 255).cpu().numpy()
    img1 = np.transpose(img1, (1, 2, 0))
    img2 = img2.data.squeeze().float().clamp(0, 255).cpu().numpy()
    img2 = np.transpose(img2, (1, 2, 0))

    img1_y = np.dot(img1, [65.738, 129.057, 25.064]) / 255.0 + 16.0
    img2_y = np.dot(img2, [65.738, 129.057, 25.064]) / 255.0 + 16.0

    if not img1_y.shape == img2_y.shape:
        raise ValueError('Input images must have the same dimensions.')

    h, w = img1_y.shape[:2]
    if border > 0:
        img1_y = img1_y[border:h - border, border:w - border]
        img2_y = img2_y[border:h - border, border:w - border]

    return ssim(img1_y, img2_y)


def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    kernel_1d = windows.gaussian(11, std=1.5)
    kernel    = np.outer(kernel_1d, kernel_1d)
    kernel    = kernel / np.sum(kernel)

    mu1 = convolve2d(img1, kernel, mode='same', boundary='symm')
    mu2 = convolve2d(img2, kernel, mode='same', boundary='symm')

    mu1_sq  = mu1 * mu1
    mu2_sq  = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = convolve2d(img1 * img1, kernel, mode='same', boundary='symm') - mu1_sq
    sigma2_sq = convolve2d(img2 * img2, kernel, mode='same', boundary='symm') - mu2_sq
    sigma12   = convolve2d(img1 * img2, kernel, mode='same', boundary='symm') - mu1_mu2

    numerator   = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map    = numerator / (denominator + 1e-12)

    return ssim_map.mean()


def make_optimizer(args, target):
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = tuple(args.betas)
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class  = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            """
            epoch: رقم الـ epoch الذي سنبدأ منه (مثلاً 131)
            - لا نحمّل optimizer.pt القديم، بل نبدأ optimizer جديد ونحرك scheduler
              بالعدد المناسب من الخطوات.
            """
            if epoch > 1:
                steps = epoch - 1
                print(f'Applying {steps} scheduler steps to set last_epoch to {steps}.')
                for _ in range(steps):
                    self.scheduler.step()

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch + 1

    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer