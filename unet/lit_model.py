import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
from torch import optim
import pytorch_lightning as pl
import numpy as np
import os
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

from pytorch_lightning.metrics import Recall

from utils import *
import cv2
class LitLambdaUnet(pl.LightningModule):
    def __init__(self, cfg, model):
        super(LitLambdaUnet, self).__init__()
        self.model = model
        self.criterion = nn.BCEWithLogitsLoss()
        self.out = nn.Sigmoid()
        self.lambdaLayer = cfg.lambdaLayer
        self.cut_marginal = cfg.cut_marginal
        self.lr_scheduler = cfg.lr_scheduler
        self.temporal = cfg.temporal
        self.val_train = cfg.val_train
        self.cfg = cfg

    def training_step(self, batch, batch_idx):
        imgs, true_masks = batch
        self.cfg.var_t = imgs.shape[1]
        imgs = rearrange(imgs, 'b t c hh ww -> (b t) c hh ww')

        masks_pred = self.model(imgs)
        masks_pred_b = self.out(masks_pred)
        masks_pred_b = (masks_pred_b > 0.5).float()

        if self.cut_marginal and self.lambdaLayer and self.temporal:
            masks_pred, masks_pred_b, true_masks = cut_marginal(masks_pred, masks_pred_b, true_masks)

        true_masks = rearrange(true_masks, 'b t c hh ww -> (b t) c hh ww')

        loss = self.criterion(masks_pred, true_masks)
        dice = dice_coeff(masks_pred_b, true_masks).item()
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        self.log('train_dice', dice, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.val_train:
            imgs, true_masks = batch
        else:
            imgs, true_masks, this_len = batch
        self.cfg.var_t = imgs.shape[1]
        imgs = rearrange(imgs, 'b t c hh ww -> (b t) c hh ww')

        masks_pred = self.model(imgs)
        masks_pred_b = self.out(masks_pred)
        masks_pred_b = (masks_pred_b > 0.5).float()

        if self.cut_marginal and self.lambdaLayer and self.val_train and self.temporal:
            masks_pred, masks_pred_b, true_masks = cut_marginal(masks_pred, masks_pred_b, true_masks)

        true_masks = rearrange(true_masks, 'b t c hh ww -> (b t) c hh ww')

        if not self.val_train:
            this_len = this_len.cpu().numpy()[0]
            masks_pred, masks_pred_b, true_masks = masks_pred[:this_len, ...], masks_pred_b[:this_len, ...], true_masks[:this_len, ...]

        if self.cfg.sig:
            sig = nn.Sigmoid()
            masks_pred_sig = sig(masks_pred)
            cri = nn.BCELoss()
            loss = cri(masks_pred_sig,true_masks) 
        else:           
            loss = self.criterion(masks_pred, true_masks)
        dice = dice_coeff(masks_pred_b, true_masks).item()

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        self.log('val_dice', dice, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # imgs, true_masks, cases = batch
        if self.cfg.step_val and not self.cfg.val_train:
            imgs, true_masks, this_len = batch
        else:
            imgs, true_masks, case = batch
        self.cfg.var_t = imgs.shape[1]
        imgs = rearrange(imgs, 'b t c hh ww -> (b t) c hh ww')
        
        masks_pred = self.model(imgs)
        masks_pred_b = self.out(masks_pred)
        masks_pred_b = (masks_pred_b > 0.5).float()

        true_masks = rearrange(true_masks, 'b t c hh ww -> (b t) c hh ww')

        if self.cfg.step_val and not self.cfg.val_train:
            this_len = this_len.cpu().numpy()[0]
            masks_pred, masks_pred_b, true_masks = masks_pred[:this_len, ...], masks_pred_b[:this_len, ...], true_masks[:this_len, ...]
          

        if self.cfg.sig:
            sig = nn.Sigmoid()
            masks_pred_sig = sig(masks_pred)
            cri = nn.BCELoss()
            loss = cri(masks_pred_sig,true_masks) 
        else:           
            loss = self.criterion(masks_pred, true_masks)
        dice = dice_coeff(masks_pred_b, true_masks).item()

        # cmpt_recall = Recall(num_classes=1)
        # recall = cmpt_recall(true_masks, masks_pred_b)
        y_true = true_masks.cpu().numpy().ravel()
        y_pred = masks_pred_b.cpu().numpy().ravel()
        if y_true.max() == 0 and y_pred.max() == 0:
            recall = 1.0
            prec = 1.0
            f1 = 1.0
        else:
            recall = recall_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

        self.log('test_loss', loss)
        self.log('test_dice', dice)
        self.log('test_recall', recall)
        self.log('test_prec', prec)
        self.log('test_f1', f1)

        # masks_pred_b = rearrange(masks_pred_b, '(b t) c hh ww -> b t c hh ww', b=batch[0].shape[0])
        case_dir = f'{self.cfg.res_dir}/{case[0]}'
        os.makedirs(case_dir, exist_ok=True)
        for t in range(masks_pred_b.shape[0]):
            pred_map = masks_pred_b[t,...]
            target =true_masks[t,...] 
            result = overlay(target, pred_map)
            out_file = f'{case_dir}/{t:02d}.png'
            cv2.imwrite(out_file, result)


    def configure_optimizers(self):
        optimizer = optim.RMSprop(self.model.parameters(), lr=0.001, weight_decay=1e-8, momentum=0.9)
        if self.lr_scheduler: 
            return {
                'optimizer': optimizer,
                # 'lr_scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5),
                # 'lr_scheduler': optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1),
                'lr_scheduler': get_scheduler(optimizer, 'lambda', nepoch_fix=20, nepoch=200),
                'monitor': 'val_dice'
            }
        else:
            return optimizer
