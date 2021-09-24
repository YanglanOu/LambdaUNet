import os
import argparse
from einops import rearrange

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from numpy.random import RandomState

from unet import UNet, ThreeDUNet, AttnUNet
from unet.lit_model import LitLambdaUnet
from unet.lit_3D_unet import Lit3DUnet
from unet.trans_unet import VisionTransformer, TRANSCONFIG
from lit_dataset import Stroke_lambda_test
from config import Config
from utils import *

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg', dest='cfg', type=str, default='test',
                        help="config file.")
    parser.add_argument('-f', '--fold', dest='fold', type=int, default=0,
                        help="fold.")
    parser.add_argument('-v', '--version', dest='version', type=int, default=0,
                        help="version.")
    parser.add_argument('-e', '--epoch', dest='epoch', type=int, default=-1,
                        help="best epoch.")                           
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    cfg = Config(args.cfg, args.fold, False)
    rs = RandomState(523)

    data_dir = os.path.expanduser('~/data/Stroke_AWS_DWI_Anom_PSU/')
    training_cases = 'training_cases.txt'

    np.random.seed(cfg.seed) 
    torch.manual_seed(cfg.seed)

    indx = list(range(32))
    rs.shuffle(indx)

    if args.epoch > 0:
        model_name = f'lambdaunet-epoch={args.epoch}'
    else:
        model_name = 'lambdaunet-best'

    if args.fold == 0:
        val_indx = indx[:20]
        test_indx = indx[20:]
    elif args.fold == 1:
        val_indx = indx[:10] + indx[20:]
        test_indx = indx[10:20]
    elif args.fold == 2:
        val_indx = indx[10:]
        test_indx = indx[:10]
    else:
        print("invalid fold")
        exit(0)

    test_dataset = Stroke_lambda_test(data_dir, training_cases, test_indx, cfg.seq_len)
 
    testloader = DataLoader(test_dataset,
                batch_size=1, shuffle=False, num_workers=0, drop_last=True, pin_memory=True) 
                
    if cfg.three_D_Unet:
        net = ThreeDUNet(cfg)
        net = Lit3DUnet.load_from_checkpoint(f'/home/yxo43/results/stroke/LambdaUnet/pl/{args.cfg}/{args.fold}/lightning_logs/version_{args.version}/checkpoints/lambdaunet-best.ckpt', model=net, cfg=cfg, fold=args.fold)
    elif cfg.attn_UNet:
        net = AttnUNet(cfg)
        net = LitLambdaUnet.load_from_checkpoint(f'/home/yxo43/results/stroke/LambdaUnet/pl/{args.cfg}/{args.fold}/lightning_logs/version_{args.version}/checkpoints/lambdaunet-best.ckpt', model=net, cfg=cfg, fold=args.fold)
    elif cfg.trans_Unet is not None:
        trans_config = TRANSCONFIG[cfg.trans_Unet]
        net = VisionTransformer(trans_config, img_size=256, num_classes=1)
        net = LitLambdaUnet.load_from_checkpoint(f'/home/yxo43/results/stroke/LambdaUnet/pl/{args.cfg}/{args.fold}/lightning_logs/version_{args.version}/checkpoints/lambdaunet-best.ckpt', model=net, cfg=cfg, fold=args.fold)
    else:
        net = UNet(cfg)
        net = LitLambdaUnet.load_from_checkpoint(f'/home/yxo43/results/stroke/LambdaUnet/pl/{args.cfg}/{args.fold}/lightning_logs/version_{args.version}/checkpoints/lambdaunet-best.ckpt', model=net, cfg=cfg, fold=args.fold)

    trainer = pl.Trainer(gpus=[1], default_root_dir=cfg.test_dir)
    log = trainer.test(net, test_dataloaders=testloader)
    dice = log[0]['test_dice']
    recall = log[0]['test_recall']
    prec = log[0]['test_prec']
    f1 = log[0]['test_f1']

    write_metrics_to_csv('results.csv', cfg, args.fold, dice, recall, prec, f1)

    # checkpoint = pl_load(checkpoint_path)
