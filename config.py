import yaml
import os
import numpy as np
import shutil


def recreate_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)


class Config:
    def __init__(self, cfg_id, fold, create_dirs=True):
        self.id = cfg_id
        self.fold = fold
        cfg_name = 'configs/%s.yml' % cfg_id
        if not os.path.exists(cfg_name):
            print("Config file doesn't exist: %s" % cfg_name)
            exit(0)
        with open(cfg_name, 'r') as stream:
            cfg = yaml.safe_load(stream)

        # res dir for pytorch lighting 
        self.base_dir = os.path.expanduser(cfg['results_dir'])

        self.cfg_dir = '%s/%s/%s' % (self.base_dir, cfg_id, fold)
        self.res_dir = '%s/%s/results' % (self.base_dir, cfg_id)

        self.test_dir = '%s/test' % self.cfg_dir

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.res_dir, exist_ok=True)

        # param
        self.gpus = cfg['gpus']
        self.n_channels = cfg['n_channels']
        self.n_classes = cfg['n_classes']
        self.shuffle = cfg['shuffle']

        self.lr = cfg['learning_rate']
        self.batchsize = cfg['batchsize']
        self.val_batchsize = cfg['val_batchsize']
        self.seq_len = cfg['seq_len']
        self.bilinear = cfg['bilinear']
        self.patience = cfg['patience']

        self.lambdaLayer = cfg['lambdaLayer'] 
        self.sig = cfg.get('sigmoid',False)
        self.uplambdaLayer = cfg.get('uplambdaLayer',[False, False, False, False])
        self.temporal = cfg['temporal']
        self.cut_marginal = cfg['cut_marginal']
        self.lr_scheduler = cfg['lr_scheduler']
        self.val_train = cfg['val_train']
        self.kernel_size = cfg['kernel_size']
        self.seed = cfg['seed']
        self.norm = cfg.get('norm', 'BatchNorm2d')
        self.tr = cfg.get('tr', 3)

        self.step_val = cfg.get('step_val', False)
        self.var_t = cfg['var_t']

