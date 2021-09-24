from einops import rearrange
import numpy as np
import pandas as pd

import torch
from torch.autograd import Function
from torch.optim import lr_scheduler

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def diff(array1, array2):
    array1 = array1.cpu().numpy()
    array2 = array2.cpu().numpy()
    diff = sum(sum(sum(abs(array1 - array2))))
    equal = np.array_equal(array1, array2)
    return diff, equal

def cut_marginal(masks_pred, masks_pred_b, true_masks):
        masks_pred = rearrange(masks_pred, '(b t) c hh ww -> b t c hh ww', b=true_masks.shape[0])
        masks_pred = masks_pred[:,1:-1, ...]
        masks_pred = rearrange(masks_pred, 'b t c hh ww -> (b t) c hh ww')

        masks_pred_b = rearrange(masks_pred_b, '(b t) c hh ww -> b t c hh ww', b=true_masks.shape[0])
        masks_pred_b = masks_pred_b[:,1:-1, ...]
        masks_pred_b = rearrange(masks_pred_b, 'b t c hh ww -> (b t) c hh ww')

        true_masks = true_masks[:,1:-1, ...]

        return masks_pred, masks_pred_b, true_masks

def write_metrics_to_csv(csv_file, cfg, fold, dice, recall, precision, f1):
    df = pd.read_csv(csv_file)
    if not ((df['config_id'] == cfg.id) & (df['fold'] == fold)).any():
        df = df.append({'config_id': cfg.id, 'fold': fold}, ignore_index=True)
    index = (df['config_id'] == cfg.id) & (df['fold'] == fold)
    df.loc[index, 'results_dir'] = cfg.cfg_dir
    df.loc[index, 'dice'] = dice
    df.loc[index, 'recall'] = recall
    df.loc[index, 'precision'] = precision     
    df.loc[index, 'f1'] = f1     

    df.to_csv(csv_file, index=False, float_format='%f')

def get_scheduler(optimizer, policy, nepoch_fix=None, nepoch=None, decay_step=None):
    if policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - nepoch_fix) / float(nepoch - nepoch_fix + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=decay_step, gamma=0.1)
    elif policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', policy)
    return scheduler


def overlay(label_i, pred_i):
    # pred_i = np.round(pred_i / 255.0)
    label_i = label_i.squeeze().cpu().numpy()
    pred_i = pred_i.squeeze().cpu().numpy()
    union = label_i * pred_i
    white = label_i - union
    red = pred_i - union

    white = np.stack((white,) * 3, axis=-1)
    white = white * [240, 240, 240]

    red = np.stack((red,) * 3, axis=-1)
    red = red * [102, 102, 255]

    union = np.stack((union,) * 3, axis=-1)
    union = union * [102, 204, 0]    

    res_fig = white + red + union

    return res_fig