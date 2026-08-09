import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import utility
import data
import model
import loss
from option import args
set_seed(args.seed)
from trainer import Trainer
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="0"
os.environ['CUDA_CACHE_PATH']='~/.cudacache'


# torch.manual_seed(args.seed)
torch.cuda.empty_cache()
checkpoint = utility.checkpoint(args)
# python /home/linhanjiang/projects/AIM/EDSR/src/main.py --model rfdn --data_test Set5+Set14+
# B100+Urban100+DIV2K  --data_range 801-900 --scale 4 --save rfdn_x4 --pre_train /home/linhanjiang/projects/AIM/EDSR/experiment/test/model/model_best.pt --rgb_range 1 --test_only --save_results
# ... الكود الأصلي في الأعلى ...

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            
            # إضافة EMA إذا طلب المستخدم
            ema = None
            if args.ema and not args.test_only:
                from trainer import ModelEMA   # سنقوم بتعريفه في trainer.py
                ema = ModelEMA(_model, args.ema_decay)
            
            t = Trainer(args, loader, _model, _loss, checkpoint, ema=ema)   # تمرير ema إلى Trainer
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()