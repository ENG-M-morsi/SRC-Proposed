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
# import cv2

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        # إنشاء optimizer
        self.optimizer = utility.make_optimizer(args, self.model)

        # ⚠️ لا تحمّل optimizer إلا لو الملف موجود ويطابق
        if self.args.load != '':
            opt_path = os.path.join(ckp.dir, 'optimizer.pt')
            if os.path.isfile(opt_path):
                try:
                    self.optimizer.load(ckp.dir, epoch=len(ckp.log))
                    self.ckp.write_log('Optimizer state loaded successfully.')
                except ValueError:
                    self.ckp.write_log(
                        'WARNING: Optimizer state is incompatible with current model. '
                        'Optimizer has been re-initialized.'
                    )

            else:
                self.ckp.write_log('No optimizer checkpoint found, training with fresh optimizer.')
    
        # إنشاء optimizer
        self.error_last = 1e8

        # ------------------------------------------
        # عرض النموذج وعدد المعاملات
        print(self.model)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')

        total_trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f'{total_trainable_params:,} training parameters.')

        # تسجيل نفس المعلومات في log.txt
        self.ckp.write_log(str(self.model))
        self.ckp.write_log(f'{total_params:,} total parameters.')
        self.ckp.write_log(f'{total_trainable_params:,} training parameters.')

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch()
        lr = self.optimizer.get_lr()

        # تسجيل رقم الإيبوك ومعدل التعلم
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )

        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()

        # TEMP
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

        # إنهاء تسجيل الـ loss لهذا الـ epoch
        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

        # حساب المتوسط العام لقيم الـ loss لهذا الـ epoch
        avg_loss = self.loss.log.mean()
        self.ckp.write_log(f'Average Loss: {avg_loss:.4f}')

        # --- إضافة التقييم بعد كل Epoch ---
        if not self.args.test_only:  # فقط أثناء التدريب
            self.ckp.write_log(f'\n--- Running Evaluation at Epoch {epoch + 1} ---')
            self.test()

    def test(self):
        # منع التكرار داخل نفس الجلسة
        if hasattr(self, 'last_test_epoch') and self.last_test_epoch == self.optimizer.get_last_epoch():
            return
        self.last_test_epoch = self.optimizer.get_last_epoch()

        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')

        # add_log creates a new row/entry for PSNR log (existing behavior)
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))

        # Create or expand SSIM log
        if not hasattr(self.ckp, 'log_ssim'):
            self.ckp.log_ssim = torch.zeros_like(self.ckp.log)
        else:
            if self.ckp.log_ssim.shape[0] < self.ckp.log.shape[0]:
                extra = self.ckp.log.shape[0] - self.ckp.log_ssim.shape[0]
                self.ckp.log_ssim = torch.cat(
                    [self.ckp.log_ssim, torch.zeros(extra, self.ckp.log.shape[1], self.ckp.log.shape[2])],
                    dim=0
                )

        self.model.eval()
        timer_test = utility.timer()

        if self.args.save_results:
            self.ckp.begin_background()

        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                eval_acc, eval_acc_ssim = 0.0, 0.0
                PSNR_values, SSIM_values, image_names = [], [], []

                for lr, hr, filename in tqdm(d, ncols=80):
                    image_names.append(filename[0])
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)
                    save_list = [sr]

                    psnr = utility.calc_psnr(sr, hr, scale, self.args.rgb_range, dataset=d)
                    ssim_val = utility.calc_ssim(sr, hr, scale, self.args.rgb_range, dataset=d)

                    eval_acc += psnr
                    eval_acc_ssim += ssim_val
                    PSNR_values.append(psnr)
                    SSIM_values.append(ssim_val)

                    self.ckp.log[-1, idx_data, idx_scale] += psnr
                    self.ckp.log_ssim[-1, idx_data, idx_scale] += ssim_val

                    if self.args.save_gt:
                        save_list.extend([lr, hr])
                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                # الحساب النهائي للمتوسط لهذه الـ dataset و scale
                n_images = len(d)
                avg_psnr = eval_acc / max(1, n_images)
                avg_ssim = eval_acc_ssim / max(1, n_images)

                # finalize logs (average per-image)
                self.ckp.log[-1, idx_data, idx_scale] /= max(1, n_images)
                self.ckp.log_ssim[-1, idx_data, idx_scale] /= max(1, n_images)

                # append 'average' row to Excel
                image_names.append('average')
                PSNR_values.append(avg_psnr)
                SSIM_values.append(avg_ssim)

                # save results to Excel
                if self.args.load in ('.', ''):
                    xlsx_file_name = f'../experiment/{self.args.save}/results-{d.dataset.name}/{d.dataset.name}.xlsx'
                else:
                    xlsx_file_name = f'../experiment/{self.args.load}/results-{d.dataset.name}/{d.dataset.name}.xlsx'

                os.makedirs(os.path.dirname(xlsx_file_name), exist_ok=True)
                data = {
                    'image': np.array(image_names, dtype=object),
                    'psnr': [round(i, 2) for i in PSNR_values],
                    'ssim': [round(i, 4) for i in SSIM_values]
                }
                df = DataFrame(data)
                df.to_excel(xlsx_file_name)

                # احسب أفضل القيم عبر كل العصور (epochs)
                best_psnr_vals = self.ckp.log.max(0)
                best_psnr = float(best_psnr_vals[0][idx_data, idx_scale].item())
                best_psnr_epoch = int(best_psnr_vals[1][idx_data, idx_scale].item()) + 1

                best_ssim_vals = self.ckp.log_ssim.max(0)
                best_ssim = float(best_ssim_vals[0][idx_data, idx_scale].item())
                best_ssim_epoch = int(best_ssim_vals[1][idx_data, idx_scale].item()) + 1

                # ✅ الطباعة الصحيحة في الموضع المناسب داخل الحلقة
                if self.args.test_only:
                    # حالة الاختبار فقط: بدون رقم epoch أو أفضل قيم
                    log = '[{} x{}]\tAverage PSNR: {:.3f} dB | SSIM: {:.4f}'.format(
                        d.dataset.name, scale, avg_psnr, avg_ssim)
                else:
                    # حالة التدريب/التحقق: أظهر رقم epoch وأفضل القيم
                    log = '[{} x{}]\tEpoch {}: Average PSNR: {:.3f} dB | SSIM: {:.4f}  (Best PSNR: {:.3f} @epoch {}, Best SSIM: {:.4f} @epoch {})'.format(
                        d.dataset.name, scale, epoch, avg_psnr, avg_ssim, best_psnr, best_psnr_epoch, best_ssim, best_ssim_epoch)

                self.ckp.write_log(log)

        # نهاية الحلقة الكبرى
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best_psnr_epoch == epoch))

        self.ckp.write_log('Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True)
        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch()
            return epoch >= self.args.epochs
