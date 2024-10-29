import sys
from pathlib import Path

sys.path.append('.')
sys.path.insert(0, str(Path(__file__).resolve().parent) + '/data_compression/data_compression')

import argparse
import datetime
import glob
import os
import random

import cv2
import numpy as np
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR
from torch.utils.data import DataLoader, Dataset

from data_compression.datasets.datasets import ImgLMDBDataset, data_prefetcher
from models.utils.mylib import (generate_local_region_msk,
                                generate_random_group_msk, load_img, write_log)
from models.utils.pytorch_msssim import MSSSIMLoss, ms_ssim

torch.backends.cudnn.benchmark=True
torch.set_num_threads(1)
# reproducibility
seed_num = 3407
torch.manual_seed(seed_num)
random.seed(seed_num)
np.random.seed(seed_num)

# Common Setting
parser = argparse.ArgumentParser()
parser.add_argument('--eval-only', action='store_true')
parser.add_argument('--model', type=str, default='ours_meanscalehyper')
parser.add_argument('--total-iteration', type=int, default=2000000)
parser.add_argument('--saving-iteration', type=int, default=0)
parser.add_argument('--eval-interval', type=int, default=100)
parser.add_argument('--saving-interval', type=int, default=0)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--num-workers', type=int, default=8)
parser.add_argument('--lmbda', type=int, default=1024)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--N', type=int, default=128)
parser.add_argument('--M', type=int, default=192)
parser.add_argument('--char_layernum', type=int, default=3)
parser.add_argument('--char_embed', type=int, default=128)
parser.add_argument("--transform-channels", type=int, nargs='+',
                    default=[128, 128, 128, 192],help="Transform channels.")
parser.add_argument("--hyper-channels", type=int, nargs='+',
                    default=None,help="Transform channels.")
parser.add_argument("--depths_char", type=int, nargs='+',
                    default=[1,1,1],help="Depth of GroupSwinBlocks in ChARM.")
parser.add_argument("--num_heads_char", type=int, nargs='+',
                    default=[8,8,8],help="Head num of GroupSwinBlocks in ChARM.")
parser.add_argument('--patch-size', type=int, default=256)
parser.add_argument('--train-set', type=str, default='/data1/datasets/Imagenet')
parser.add_argument('--eval-set', type=str, default='/data1/datasets/kodak')
parser.add_argument('--eval-folders', action='store_true', 
    help='there are folders in the args.eval_set')
parser.add_argument('--save', '-s', default='./logs/default', type=str, help='directory for saving')
parser.add_argument('--metric', type=str, nargs='+',
                    default=['mse'], choices=['mse', 'msssim', 'lpips'])
parser.add_argument('--scheduler', type=str, default='multistep')
parser.add_argument('--multistep-milestones', type=int, nargs='+', default=[1800000])
parser.add_argument('--multistep-gamma', type=float, default=0.1)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--save-result', type=str, default='')
parser.add_argument('--save-qmap', type=str, default='')
parser.add_argument('--reset-rdo', action='store_true', help='reset the rdo to +inf.')
parser.add_argument('--soft-then-hard', action='store_true')
parser.add_argument('--soft-then-hard-start-iteration', type=int, default=0)
parser.add_argument('--freeze-transform', action='store_true')
parser.add_argument('--start-joint-training-iteration', type=int, default=-1)

# Transformer based Transform Coding
parser.add_argument('--swin-disable-norm', action='store_true',
                    help='do not use any normalization in the transformation.')

# GroupViT
parser.add_argument('--only-rec-fg', action='store_true')
parser.add_argument('--groupvit-save-group-msk', type=str, default='')
parser.add_argument('--groupvit-load-group-msk', type=str, default='figs/kodak/group_mask')
parser.add_argument("--groups_tobe_decode", type=int, nargs='+',
                    default=[0, 0, 0], help="group idxs to be decoded.")

# Analysis Setting
parser.add_argument('--visualize-bit-allocation', type=str, default='',
                    help='path to the bit allocation saving directory.')

# ddp training
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

args = parser.parse_args()

import our_codecs

models = {
    'ours_groupswin_channelar': our_codecs.GroupChARTTC,
    'ours_groupswin_channelar_woStructure': our_codecs.GroupChARTTC_woStructure,
}


class DefaultTrainer():
    def __init__(self):
        self.build_logger()
        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.train_loader = self.build_train_loader()
        self.eval_ps = self.load_eval_ps()
        self.loss_fn = self.build_loss_fn()
        
        if args.resume:
            self.resume(args.resume)
        else:
            self.best_rdo = float('+inf')
            self.start_iteration = 1

    def build_logger(self):
        assert len(args.metric) == 1
        args.metric = args.metric[0]
        if args.metric == 'msssim':
            args.save = os.path.join(args.save, args.model+'_msssim', str(args.lmbda))
        elif args.model == 'ours_groupswin_TfChARM':   
            args.save = os.path.join(args.save, args.model, str(args.lmbda),
                                     f'ChAR-Layer{args.char_layernum}' +
                                     f'Embed{args.char_embed}' +
                                     'Depth'+str(''.join(str(t) for t in args.depths_char)) +
                                     'NumHeads'+str(''.join(str(t) for t in args.num_heads_char)))

        os.makedirs(args.save, exist_ok=True)
        self.p_log = os.path.join(
            args.save,
            '{}.txt'.format(str(datetime.datetime.now()).replace(':', '-')[:-7]))
        write_log(self.p_log, str(args).replace(', ', ',\n\t') + '\n')
        
    def log(self, content):
        return write_log(self.p_log, content)

    def ddp_training(self):
        # todo : code of this part is unusable. Debug it in the future. 
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        # For multiprocessing distributed training, rank needs to be the
        # global rank among all the processes
        ngpus_per_node = torch.cuda.device_count()
        args.rank = args.rank * ngpus_per_node 

        torch.distributed.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        self.model = torch.nn.parallel.DistributedDataParallel(
            # self.model, find_unused_parameters=True)
            self.model)

    def build_model(self):
        if 'ours' in args.model:
            if 'swin' in args.model:
                if args.swin_disable_norm:
                    norm_layer = nn.Identity
                    self.log('disable the layer normalization in transformation.')
                else:
                    norm_layer = nn.LayerNorm
                if 'TfChARM' in args.model:
                    model = models[args.model](norm_layer=norm_layer,
                                               char_layernum=args.char_layernum,
                                               depths_char=args.depths_char,
                                               num_heads_char=args.num_heads_char,
                                               char_embed=args.char_embed)
                else:
                    model = models[args.model](norm_layer=norm_layer)
            elif 'group' in args.model:
                model = models[args.model](args.hyper_channels)
            else:
                model = models[args.model](args.transform_channels, args.hyper_channels)
        elif 'elic' in args.model:
            model = models[args.model]()
        else:
            model = models[args.model](args.N, args.M)
        model.train()
        model.cuda()
        self.log('\n'+str(model)+'\n\n')
        model.forward_return_dict = True       # used to enable training
        return model

    def build_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        return optimizer

    def build_scheduler(self):
        assert args.scheduler in ['multistep', 'cos']
        if args.scheduler == 'multistep':
            scheduler = MultiStepLR(
                self.optimizer, 
                milestones=args.multistep_milestones, 
                gamma=args.multistep_gamma)
        elif args.scheduler == 'cos':
            scheduler = CosineAnnealingLR(self.optimizer, args.total_iteration)
        else:
            raise NotImplementedError
        self.log('scheduler: {}\n'.format(scheduler))
        
        return scheduler

    def build_train_loader(self):

        if args.train_set.endswith('datasets/COCO/train2017'):      # hard code for debugging
            class UnlabeledImageDataset(Dataset):
                def __init__(self, root_dir, patch_size, transform=None):
                    """
                    初始化数据集

                    参数：
                        root_dir (str): 图片所在的根目录
                        patch_size (int): 图片裁剪的补丁大小
                        transform (callable, optional): 对图像进行的变换
                    """
                    self.root_dir = os.path.expanduser(root_dir)
                    self.image_paths = [
                        os.path.join(self.root_dir, fname) 
                        for fname in os.listdir(self.root_dir) 
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
                    ]
                    self.transform = transform
                    self.patch_size = patch_size

                def __len__(self):
                    return len(self.image_paths)

                def __getitem__(self, idx):
                    img_path = self.image_paths[idx]
                    image = Image.open(img_path).convert('RGB')
                    # check if the image is grayscale, if so convert to 3 channel
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    # check if the short edge is less than patch_size, if so, skip this image
                    if min(image.size) < self.patch_size:
                        return self.__getitem__(random.randint(0, len(self)-1))

                    if self.transform:
                        image = self.transform(image)
                    
                    return image

            transform = transforms.Compose([
                transforms.RandomCrop(args.patch_size),  # 随机裁剪到指定大小
                transforms.ToTensor(),                  # 转换为张量
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # 根据需要调整均值和标准差
                #                     std=[0.229, 0.224, 0.225])
            ])
            dataset = UnlabeledImageDataset(
                root_dir=args.train_set,
                patch_size=args.patch_size,
                transform=transform
            )
            # 填充代码结束
        else:
            # previous code
            if args.lmbda > 512:
                args.train_set = '/data1/datasets/Flickr2K_HR_lmdb'
            else:
                args.train_set = '/data1/datasets/Coco'
            self.log('training dataset path: {}\n'.format(
                args.train_set
            ))

            dataset = ImgLMDBDataset(args.train_set, is_training=True, patch_size=args.patch_size)
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True
        )
        return train_loader

    def load_eval_ps(self):
        eval_ps = sorted(glob.glob(os.path.join(args.eval_set, '*.png')))
        if eval_ps == []:
            eval_ps = sorted(glob.glob(os.path.join(args.eval_set, '*.jpg')))
        return eval_ps

    def build_loss_fn(self):
        if args.metric == 'mse':
            loss_fn = nn.MSELoss().cuda()
        elif args.metric == 'msssim':
            loss_fn = MSSSIMLoss(1.0, True).cuda()
        # loss_fn = {}
        # for metric in args.metric:
        #     if metric == 'mse':
        #         loss_fn[metric] = nn.MSELoss().cuda()
        #     if metric == 'msssim':
        #         loss_fn[metric] = MSSSIMLoss(1.0, True).cuda()
        #     if metric == 'lpips':
        #         raise NotImplementedError
        return loss_fn

    def train(self):
        self.log('pre evaluation on entire images:\n')
        self.eval()

        if 'group' in args.model:
            # print('pre evaluation on partial images:\n')
            # self.eval_partial() # Note: only for debugging
            self.eval(eval_fg=True)
            self.log('\n')

        prefetcher = data_prefetcher(self.train_loader)

        self.model.train()
        for iteration in range(self.start_iteration, args.total_iteration + 1):
            #fetch data
            frames = prefetcher.next()
            if frames is None:
                prefetcher = data_prefetcher(self.train_loader)
                frames = prefetcher.next()

            # train one step
            with torch.autograd.set_detect_anomaly(True):
                if 'group' in args.model:
                    b,c,h,w = frames[0].shape
                    msk = generate_random_group_msk(b,h,w,16)
                    for bi in range(b):
                        if random.random() > 0.5:
                            msk[bi, ...] = 0
                    
                    if not args.soft_then_hard:
                        res = self.model(frames[0], noisy=True, msk=msk)
                    else:
                        if iteration > args.soft_then_hard_start_iteration:
                            res = self.model(frames[0], noisy=False, msk=msk)
                else:
                    if not args.soft_then_hard:
                        res = self.model(frames[0], noisy=True)
                    else:
                        if iteration > args.soft_then_hard_start_iteration:
                            res = self.model(frames[0], noisy=False)

            ## calculate loss
            loss = self.calculate_loss(frames[0], res)

            # optimize
            self.optimizer.zero_grad()
            loss['loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            if iteration < (args.total_iteration * 0.9):
                eval_interval = args.eval_interval * 10
            else:
                eval_interval = args.eval_interval
            # eval_interval = args.eval_interval      # ! debug

            if iteration % eval_interval == 0:
                rdo = self.eval(iteration)
                if 'group' in args.model:
                    rdo += self.eval(iteration, eval_fg=True)
                    self.log('\n')

                ## save best model
                if rdo < self.best_rdo:
                    self.best_rdo = rdo
                    self.log('Best model. Rdo is {:.4f} and save model to {}\n\n'.format(
                        rdo, args.save))
                    if iteration >= args.saving_iteration:
                        self.save_ckpt(iteration)

            if args.saving_interval:
                if (iteration+1) % args.saving_interval == 0:
                    self.log('Save model. Rdo is {:.4f} and save model to {}\n\n'.format(
                        rdo, args.save))
                    self.save_ckpt(iteration, '{}.pth'.format(iteration+1))

            if args.soft_then_hard:
                if args.soft_then_hard_start_iteration == iteration:
                    self.log('-------------------------------\n')
                    self.log('Start hard training ! \n')
                    self.log('-------------------------------\n')
                    self.model.soft_then_hard()
                    self.show_learnable_params()

            if iteration == args.start_joint_training_iteration:
                self.log('-------------------------------\n')
                self.log('Start joint training ! \n')
                self.log('-------------------------------\n')
                for p in self.model.parameters():
                    p.requires_grad = True
                self.show_learnable_params()

    def eval(self, iteration=None, eval_fg=False):
        self.model.eval()
        torch.cuda.empty_cache()
        log = {
            'bpp':0,
            'bpp_y':0,
            'bpp_side':0,
            'psnr': 0,
            'ms_ssim':0,
        }
        
        with torch.no_grad():
            for input_p in self.eval_ps:
                torch.cuda.empty_cache()
                ## forward
                x, hx, wx = load_img(input_p, padding=True, factor=64)
                x = x.cuda()

                if eval_fg:
                    if args.groupvit_load_group_msk:
                        msk = Image.open(os.path.join(args.groupvit_load_group_msk, input_p.split('/')[-1])).convert('L')
                        msk = np.asarray(msk).astype(np.uint8)
                        msk[msk!=0] = 1
                        _, msk, _, _ = cv2.connectedComponentsWithStats(
                            msk, connectivity=4)
                        msk = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0)
                    else:
                        b,c,h,w = x.shape
                        msk = generate_random_group_msk(b,h,w,16)
                        if args.groupvit_save_group_msk:
                            os.makegdirs(args.groupvit_save_group_msk, exist_ok=True)
                            torchvision.utils.save_image(
                                msk.float()/msk.max(), os.path.join(
                                args.groupvit_save_group_msk, input_p.split('/')[-1]))
                    res = self.model(x, noisy=False, msk=msk, only_rec_fg=eval_fg)
                else:
                    if 'group' in args.model:
                        b,c,h,w = x.shape
                        msk = generate_random_group_msk(b,h,w,16)
                        res = self.model(x, noisy=False, msk=msk)
                    else:
                        res = self.model(x, noisy=False)
                loss = self.calculate_loss(x, res)

                x = x[:, :, :hx, :wx].mul(255).round().clamp(0, 255)
                x_hat = res['x_hat'][:, :, :hx, :wx].mul(255).round().clamp(0, 255)
                if eval_fg:
                    fg_msk = msk.float()
                    fg_msk[fg_msk!=0] = 1
                    fg_msk = F.interpolate(fg_msk, size=(hx, wx), mode='nearest').cuda() # [1,1,hx,wx]
                    # psnr = 20 * np.log10(255.) - 10 * torch.log10((((x - x_hat) ** 2)*fg_msk).sum() / (hx* wx))
                    # print(hx, wx, hx*wx, fg_msk.shape, fg_msk.sum(), ((x - x_hat) ** 2).shape)
                    # print(torch.log10((((x - x_hat) ** 2)*fg_msk).sum() / fg_msk.sum()))
                    # print(torch.log10((((x - x_hat) ** 2)*fg_msk).sum() / (hx* wx)))
                    # raise
                    psnr = 20 * np.log10(255.) - 10 * torch.log10((((x - x_hat) ** 2)*fg_msk).sum() / (fg_msk.sum()*3))
                else:
                    psnr = 20 * np.log10(255.) - 10 * torch.log10(((x - x_hat) ** 2).mean())
                msssim = ms_ssim(x, x_hat, data_range=255).item()

                if args.save_result:
                    os.makedirs(args.save_result, exist_ok=True)
                    p_save = os.path.join(args.save_result, input_p.split('/')[-1][:-4]+'.png')
                    torchvision.utils.save_image(x_hat/255, p_save)
                    self.log('{} -> {}\n'.format(input_p, p_save))

                log = self.update_log(log, loss, psnr, msssim)

            for key in log.keys():
                log[key] /= len(self.eval_ps)

        self.display_log(log, iteration)
        self.model.train()

        rdo = self.calculate_rdo(log)

        return rdo

    def eval_partial(self, iteration=None):
        self.model.eval()
        torch.cuda.empty_cache()
        log = {
            'bpp': 0,
            'bpp_y': 0,
            'bpp_side': 0,
            'psnr': 0,
            'ms_ssim': 0,
        }

        with torch.no_grad():
            for input_p in self.eval_ps:
                torch.cuda.empty_cache()
                ## forward
                x, hx, wx = load_img(input_p, padding=True, factor=64)
                x = x.cuda()

                if args.groupvit_load_group_msk:
                    msk = Image.open(
                        os.path.join(args.groupvit_load_group_msk,
                                     input_p.split('/')[-1])).convert('L')
                    msk = np.asarray(msk).astype(np.uint8)
                    msk[msk != 0] = 1
                    _, msk, _, _ = cv2.connectedComponentsWithStats(
                        msk, connectivity=4)
                    group_mask = torch.from_numpy(msk).unsqueeze(0).unsqueeze(0).to(x.device)
                if args.groups_tobe_decode == [0, 0, 0]:
                    group_idxs = group_mask.unique().tolist()
                    group_idxs = group_idxs[1:]
                elif args.groups_tobe_decode is not None:
                    for group_idx in args.groups_tobe_decode:
                        assert group_idx in group_mask.unique()
                    group_idxs = sorted(args.groups_tobe_decode)
                else:
                    group_idxs = group_mask.unique().tolist()
                    if -1 in group_idxs:
                        group_idxs.remove(-1)
                res = self.model(x, noisy=False)
                loss = self.calculate_loss(x, res)

                x = x[:, :, :hx, :wx].mul(255).round().clamp(0, 255)
                x_hat = res['x_hat'][:, :, :hx, :wx].mul(255).round().clamp(0,
                                                                            255)
                # decoded region psnr
                msk_trans = torch.ones_like(group_mask).to(x.device)
                msk_trans[~torch.isin(group_mask,
                                      torch.tensor(group_idxs).to(
                                          x.device))] = 0
                msk_trans = F.interpolate(msk_trans.float(), size=(hx, wx),
                                          mode='nearest').cuda()  # [1,1,hx,wx]
                psnr = 20 * np.log10(255.) - 10 * torch.log10(
                    (((x - x_hat) ** 2) * msk_trans).sum() / (
                                msk_trans.sum() * 3))
                msssim = ms_ssim(x, x_hat, data_range=255).item()

                if args.save_result:
                    os.makedirs(args.save_result, exist_ok=True)
                    p_save = os.path.join(args.save_result,
                                          input_p.split('/')[-1][:-4] + '.png')
                    torchvision.utils.save_image(x_hat / 255, p_save)
                    self.log('{} -> {}\n'.format(input_p, p_save))

                log = self.update_log(log, loss, psnr, msssim)

            for key in log.keys():
                log[key] /= len(self.eval_ps)

        self.display_log(log, iteration)
        self.model.train()

        rdo = self.calculate_rdo(log)

        return rdo

    def calculate_rdo(self, log):
        ## calculate rdo
        if args.metric == 'mse':
            rdo = log['bpp'] + 1 / (10 ** (log['psnr'] / 10.)) * args.lmbda
        else:
            assert args.metric == 'msssim'
            rdo = log['bpp'] + (1 - log['ms_ssim']) * args.lmbda
        return rdo

    def calculate_loss(self, x, res):
        loss = {}        
        loss = self.calculate_dist_loss(x, res, loss)
        loss = self.calculate_bpp_loss(x, res, loss)
        loss['loss'] = args.lmbda * loss['dist_loss'] + loss['bpp_loss']
        return loss

    def calculate_dist_loss(self, x, res, loss):
        x_hat = res['x_hat']
        loss['dist_loss'] = self.loss_fn(x, x_hat)
        return loss

    def calculate_bpp_loss(self, x, res, loss):
        b, _, h, w = x.shape
        n_pixels = b*h*w
        loss['bpp_y'] = res['bits']['y'] / n_pixels
        if ('z' in res['bits'].keys()):
            loss['bpp_side'] = res['bits']['z'] / n_pixels
            loss['bpp_loss'] = loss['bpp_y'] + loss['bpp_side']
        else:
            loss['bpp_loss'] = loss['bpp_y']
        return loss

    def save_ckpt(self, iteration, name=None):
        if name:
            filename = name
        else:
            filename = 'best_prior_nc96.pth'
        try:
            self.model.fix_tables() ## fix cdf tables
        except:
            self.log('error occured when self.model.fix_tables()')

        if args.multiprocessing_distributed and torch.cuda.device_count() > 1:
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        torch.save({
            'best_rdo': self.best_rdo,
            'iteration': iteration,
            'parameters': state_dict
        }, os.path.join(args.save, filename))

    def resume(self, p_ckpt):
        ckpt = torch.load(p_ckpt)
        if 'best_rdo' in list(ckpt.keys()):
            self.best_rdo = ckpt['best_rdo']
        else:
            self.best_rdo = float('+inf')
            self.log('no best rdo loaded, set it as +inf.\n')

        if 'iteration' in list(ckpt.keys()):
            self.start_iteration = ckpt['iteration']
        else:
            self.start_iteration = 0
            self.log('no iteration loaded, set it as 0.\n')
        if args.reset_rdo:
            self.best_rdo = float('+inf')
            self.log('reset rdo to +inf.\n')
            self.start_iteration = 0
            self.log('reset iteration to 0.\n')
        if 'elic' in args.model:
            msg = self.model.load_state_dict(ckpt["params"], strict=False)
        else:
            msg = self.model.load_state_dict(ckpt['parameters'], strict=False)
        # self.log('resume the ckpt from : {} and the message is {}\n'.format(
        #     p_ckpt, msg
        # ))
        self.scheduler.step(self.start_iteration)
        self.log('resume info:\nbeginning lr: {:.6f}, best_rdo: {:.3f}\n\n'.format(
            self.optimizer.param_groups[0]['lr'], self.best_rdo))

    def update_log(self, log, loss, psnr, msssim):
        log['bpp'] += loss['bpp_loss'].item()
        log['bpp_y'] += loss['bpp_y'].item()
        if 'bpp_side' in loss.keys():
            log['bpp_side'] += loss['bpp_side'].item()
        log['psnr'] += psnr.item()
        log['ms_ssim'] += msssim
        return log

    def display_log(self, log, iter=None, n_blankline=1):
        if iter:
            self.log('iteration: {}\t'.format(iter))
        for k,v in log.items():
            self.log('{}: {:>6.5f}  '.format(k, v))
        for i in range(n_blankline+1):
            self.log('\n')

    def show_loss(self, loss, iteration, interval=100):
        if iteration % interval == 0:
            self.log('Loss: \t')
            for k,v in loss.items():
                self.log('{}: {:>7.6f}\t'.format(k, v))
            self.log('\n')

    def show_learnable_params(self):
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        self.log("Parameters to be updated: ")
        for each in enabled:
            self.log('\t{}\n'.format(str(each)))
        self.log('\n')
        

def main():
    trainer = DefaultTrainer()
    if args.freeze_transform:
        trainer.log('-------------------------------\n')
        trainer.log('Freeze parameters for transform! \n')
        trainer.log('-------------------------------\n')
        trainer.model.freeze_transform()
        trainer.show_learnable_params()  

    ## ddp training.
    if args.multiprocessing_distributed and torch.cuda.device_count() > 1:
        trainer.ddp_training()
        trainer.log('Use DDP training.\n\n')
    else:
        trainer.log('Training with a single process on 1 GPUs.\n\n')

    if args.eval_only: 
        log = trainer.eval(eval_fg=args.only_rec_fg)
    else:
        trainer.train()
    return 


if __name__ == '__main__':

    main()
