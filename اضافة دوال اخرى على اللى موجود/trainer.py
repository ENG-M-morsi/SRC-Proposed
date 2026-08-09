import os
import math
from decimal import Decimal
import utility
import torch
import torch.nn.utils as utils
from tqdm import tqdm
from pandas import DataFrame
import numpy as np
import pdb
import cv2

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss

        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            opt_path = os.path.join(ckp.dir, 'optimizer.pt')
            if os.path.isfile(opt_path):
                try:
                    #self.optimizer.load(ckp.dir, epoch=len(ckp.log))
                    #self.ckp.write_log('Optimizer state loaded successfully.')
                    if self.args.resume == -1:
                        resume_epoch = len(ckp.log) + 1
                    else:
                        resume_epoch = self.args.resume
                    self.optimizer.load(ckp.dir, epoch=resume_epoch)
                    self.ckp.write_log('Optimizer state loaded successfully.')
                except ValueError:
                    self.ckp.write_log(
                        'WARNING: Optimizer state incompatible. '
                        'Re-initialized optimizer.'
                    )
            else:
                self.ckp.write_log('No optimizer checkpoint found, fresh optimizer.')

        self.error_last = 1e8

        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        self.ckp.write_log(str(self.model))
        self.ckp.write_log(f'{total_params:,} total parameters.')
        self.ckp.write_log(f'{total_trainable_params:,} training parameters.')

    def train(self):
        self.loss.step()

        # get_last_epoch() = scheduler.last_epoch + 1
        # قبل schedule(): last_epoch=0 عند البداية → get_last_epoch()=1 ✓
        # قبل schedule(): last_epoch=N-1 عند resume → get_last_epoch()=N ✓
        epoch = self.optimizer.get_last_epoch()
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )

        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()

            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log(
                    '[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                        (batch + 1) * self.args.batch_size,
                        len(self.loader_train.dataset),
                        self.loss.display_loss(batch),
                        timer_model.release(),
                        timer_data.release()
                    )
                )
            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

        # schedule() يُحرّك last_epoch بـ +1
        self.optimizer.schedule()

        avg_loss = self.loss.log.mean()
        self.ckp.write_log(f'Average Loss: {avg_loss:.4f}')

    def test(self):
        torch.set_grad_enabled(False)

        # بعد schedule() في train():
        # scheduler.last_epoch زاد بـ 1 → get_last_epoch() = epoch_just_trained + 1
        # نطرح 1 للحصول على رقم الـ epoch الذي انتهى للتو
        epoch = self.optimizer.get_last_epoch() - 1

        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )

        if self.args.test_only:
            if not hasattr(self.ckp, 'log_ssim'):
                self.ckp.log_ssim = torch.zeros_like(self.ckp.log)
            else:
                if self.ckp.log_ssim.shape[0] < self.ckp.log.shape[0]:
                    extra = self.ckp.log.shape[0] - self.ckp.log_ssim.shape[0]
                    self.ckp.log_ssim = torch.cat(
                        [self.ckp.log_ssim,
                         torch.zeros(extra, self.ckp.log.shape[1], self.ckp.log.shape[2])],
                        dim=0
                    )

        self.model.eval()
        timer_test = utility.timer()

        if self.args.save_results:
            self.ckp.begin_background()

        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):

                d.dataset.set_scale(idx_scale)
                eval_acc      = 0.0
                eval_acc_ssim = 0.0
                PSNR_values, SSIM_values, image_names = [], [], []

                for lr, hr, filename in tqdm(d, ncols=80):
                    image_names.append(filename[0])
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)

                    if filename[0] in ['butterfly', 'baboon', '253027', 'img062', 'ARMS', '0828']:
                        output = sr.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                        if output.ndim == 3:
                            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
                        output = (output * 255.0).round().astype(np.uint8)
                        cv2.imwrite(filename[0] + '.png', output)

                    sr   = utility.quantize(sr, self.args.rgb_range)
                    psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    eval_acc += psnr
                    PSNR_values.append(psnr)
                    self.ckp.log[-1, idx_data, idx_scale] += psnr

                    # SSIM فقط في وضع test_only لتوفير الوقت أثناء التدريب
                    if self.args.test_only:
                        ssim = utility.calc_ssim(sr, hr, scale, self.args.rgb_range, dataset=d)
                        eval_acc_ssim += ssim
                        SSIM_values.append(ssim)
                        self.ckp.log_ssim[-1, idx_data, idx_scale] += ssim

                    save_list = [sr]
                    if self.args.save_gt:
                        save_list.extend([lr, hr])
                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                n_images = max(len(d), 1)
                self.ckp.log[-1, idx_data, idx_scale] /= n_images
                avg_psnr = eval_acc / n_images

                # Excel
                image_names.append('average')
                PSNR_values.append(avg_psnr)

                if self.args.load in ('.', ''):
                    xlsx_path = f'../experiment/{self.args.save}/results-{d.dataset.name}/{d.dataset.name}.xlsx'
                else:
                    xlsx_path = f'../experiment/{self.args.load}/results-{d.dataset.name}/{d.dataset.name}.xlsx'

                os.makedirs(os.path.dirname(xlsx_path), exist_ok=True)

                if self.args.test_only:
                    avg_ssim = eval_acc_ssim / n_images
                    self.ckp.log_ssim[-1, idx_data, idx_scale] /= n_images
                    SSIM_values.append(avg_ssim)
                    DataFrame({
                        'image': np.array(image_names, dtype=object),
                        'psnr':  [round(v, 2) for v in PSNR_values],
                        'ssim':  [round(v, 4) for v in SSIM_values]
                    }).to_excel(xlsx_path, index=False)
                else:
                    DataFrame({
                        'image': np.array(image_names, dtype=object),
                        'psnr':  [round(v, 2) for v in PSNR_values],
                    }).to_excel(xlsx_path, index=False)

                # is_best: best[1] هو index (0-based) → epoch (1-based) = best[1]+1
                best = self.ckp.log.max(0)
                best_epoch = int(best[1][idx_data, idx_scale].item()) + 1
                best_psnr  = float(best[0][idx_data, idx_scale].item())
                is_best    = (best_epoch == epoch)

                if self.args.test_only:
                    self.ckp.write_log(
                        '[{} x{}]\tAverage PSNR: {:.3f} dB | SSIM: {:.4f}'.format(
                            d.dataset.name, scale, avg_psnr, avg_ssim)
                    )
                else:
                    self.ckp.write_log(
                        '[{} x{}]\tEpoch {}: Average PSNR: {:.3f} dB  (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name, scale, epoch,
                            avg_psnr, best_psnr, best_epoch)
                    )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            # ✅ is_best صحيح الآن: best_epoch == epoch
            self.ckp.save(self, epoch, is_best=is_best)

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(device)
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            # بعد schedule(): get_last_epoch() = epoch_done + 1
            # نتوقف عندما يتجاوز epoch_done عدد الـ epochs المطلوبة
            epoch_done = self.optimizer.get_last_epoch() - 1
            return epoch_done >= self.args.epochs