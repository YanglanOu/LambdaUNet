import os
import argparse
from config import Config

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from numpy.random import RandomState

from unet import UNet
from unet.lit_model import LitLambdaUnet
from lit_dataset import Stroke_lambda, Stroke_lambda_val_train, Stroke_lambda_val_test, Stroke_lambda_test
from config import Config
from utils import *


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-c', '--cfg', dest='cfg', type=str, default='test',
                        help="config file.")
    parser.add_argument('-f', '--fold', dest='fold', type=int, default=2,
                        help="fold.")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    cfg = Config(args.cfg, args.fold, False)
    rs = RandomState(523)

    data_dir = os.path.expanduser('~/data/Stroke_AWS_DWI_Anom_PSU/')
    # data_dir = '/home/yxo43/data/stroke/Stroke_AWS_DWI_Anom_PSU_Partial_Truth/'
    training_cases = 'training_cases.txt'

    np.random.seed(cfg.seed) 
    torch.manual_seed(cfg.seed)

    indx = list(range(32))
    rs.shuffle(indx)

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

    train_dataset = Stroke_lambda(data_dir, training_cases, cfg.seq_len)
    if cfg.val_train:
        val_dataset = Stroke_lambda_val_train(data_dir, training_cases, val_indx, cfg.seq_len)
        val_batch = cfg.batchsize
    else: 
        val_dataset = Stroke_lambda_val_test(data_dir, training_cases, val_indx, cfg.seq_len, len(cfg.gpus))
        val_batch = cfg.val_batchsize
    test_dataset = Stroke_lambda_test(data_dir, training_cases, test_indx, cfg.seq_len)
 
    trainloader = DataLoader(train_dataset,
                    batch_size=cfg.batchsize, shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    valloader = DataLoader(val_dataset,
                batch_size=val_batch, shuffle=False, num_workers=0, drop_last=True, pin_memory=True) 
    testloader = DataLoader(test_dataset,
                batch_size=1, shuffle=False, num_workers=0, drop_last=True, pin_memory=True)  

    net = UNet(cfg)
    LambdaUnet = LitLambdaUnet(cfg, net)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_dice',
        # filename='lambdaunet-{epoch:02d}',
        filename='lambdaunet-{epoch:02d}',
        save_last=True,
        save_top_k=-1,
        mode='max',
    )
    checkpoint_callback_best = ModelCheckpoint(
        monitor='val_dice',
        # filename='lambdaunet-{epoch:02d}',
        filename='lambdaunet-best',
        save_top_k=1,
        mode='max',
    )
    trainer = pl.Trainer(gpus=cfg.gpus, auto_select_gpus=True, max_epochs=100, default_root_dir=cfg.cfg_dir, accelerator='dp', callbacks=[checkpoint_callback, checkpoint_callback_best])
    trainer.fit(LambdaUnet, trainloader, valloader)
    log = trainer.test(test_dataloaders=testloader)
    dice = log[0]['test_dice']
    recall = log[0]['test_recall']
    prec = log[0]['test_prec']
    f1 = log[0]['test_f1']

    write_metrics_to_csv('results.csv', cfg, args.fold, dice, recall, prec, f1)
    
