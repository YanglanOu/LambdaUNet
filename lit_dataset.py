import os
import numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import cv2
import glob
from torch.utils import data

class Stroke_lambda(Dataset):

    def __init__(self, data_dir, training_cases, seq_len):
        file = open(training_cases, "r")
        lines = file.readlines()
        self.eadcs = []
        self.dwis = []
        self.labels = []
        for line in lines:
            if line != '\n':
                line = line.split('/')
                case = data_dir + line[1] + '/' + line[2]

                eadc_file = case + '/eADC.nii.gz'
                eadc = nib.load(eadc_file).get_fdata()

                slices = eadc.shape[-1]

                dwi_file = case + '/rb1000.nii.gz'
                dwi = nib.load(dwi_file).get_fdata()

                label_file = case + '/rb1000_roi.nii.gz'
                label = nib.load(label_file).get_fdata()
                
                for i in range(slices-seq_len+1):
                    self.eadcs.append(eadc[...,i:i+seq_len])
                    self.dwis.append(dwi[...,i:i+seq_len])
                    self.labels.append(label[...,i:i+seq_len])

    def __len__(self):
        return len(self.eadcs)

    def __getitem__(self, idx):
        eadc = self.eadcs[idx]
        eadc = eadc.astype(np.float32)
        eadc = np.transpose(eadc, (2, 0, 1))
        eadc = np.expand_dims(eadc, axis=1)

        dwi = self.dwis[idx]
        dwi = dwi.astype(np.float32)
        dwi = np.transpose(dwi, (2, 0, 1))
        dwi = np.expand_dims(dwi, axis=1)
        
        input_data = np.concatenate((eadc, dwi), axis=1)
        
        label = self.labels[idx]
        label = label.astype(np.float32)
        label = np.transpose(label, (2, 0, 1))
        label = np.expand_dims(label, axis=1)

        return input_data, label


class Stroke_lambda_val_train(Dataset):
# val data has the same setting as training
    def __init__(self, data_dir, training_files, indx, seq_len):
        self.eadcs = []
        self.dwis = []
        self.labels = []
        self.seq_len = seq_len
        training_cases = []
        train_file = open(training_files, "r")
        lines = train_file.readlines()
        for line in lines:
            if line != '\n':
                line = line.split('/')
                case = data_dir + line[1] + '/' + line[2]
                training_cases.append(case)

        all_files = glob.glob(os.path.join(data_dir, '*/*'))
        all_files.sort()
        val_list = list(set(all_files) - set(training_cases))
        val_list.sort()
        for case in [val_list[i] for i in indx]:
            eadc_file = case + '/eADC.nii.gz'
            eadc = nib.load(eadc_file).get_fdata()

            slices = eadc.shape[-1]

            dwi_file = case + '/rb1000.nii.gz'
            dwi = nib.load(dwi_file).get_fdata()

            label_file = case + '/rb1000_roi.nii.gz'
            label = nib.load(label_file).get_fdata()

            for i in range(slices-seq_len+1):
                self.eadcs.append(eadc[...,i:i+seq_len])
                self.dwis.append(dwi[...,i:i+seq_len])
                self.labels.append(label[...,i:i+seq_len])

    def __len__(self):
        return len(self.eadcs)

    def __getitem__(self, idx):
        eadc = self.eadcs[idx]
        eadc = eadc.astype(np.float32)
        eadc = np.transpose(eadc, (2, 0, 1))
        eadc = np.expand_dims(eadc, axis=1)

        dwi = self.dwis[idx]
        dwi = dwi.astype(np.float32)
        dwi = np.transpose(dwi, (2, 0, 1))        
        dwi = np.expand_dims(dwi, axis=1)
        
        input_data = np.concatenate((eadc, dwi), axis=1)

        label = self.labels[idx]
        label = label.astype(np.float32)
        label = np.transpose(label, (2, 0, 1))        
        label = np.expand_dims(label, axis=1)

        return input_data, label

class Stroke_lambda_val_test(Dataset):

    def __init__(self, data_dir, training_files, indx, seq_len, num_gpus):
        self.eadcs = []
        self.dwis = []
        self.labels = []
        self.seq_len = seq_len
        self.all_len = []
        self.num_gpus = num_gpus
        training_cases = []
        train_file = open(training_files, "r")
        lines = train_file.readlines()
        for line in lines:
            if line != '\n':
                line = line.split('/')
                case = data_dir + line[1] + '/' + line[2]
                training_cases.append(case)

        all_files = glob.glob(os.path.join(data_dir, '*/*'))
        all_files.sort()
        val_list = list(set(all_files) - set(training_cases))
        val_list.sort()
        for case in [val_list[i] for i in indx]:
            eadc_file = case + '/eADC.nii.gz'
            eadc = nib.load(eadc_file).get_fdata()

            self.all_len.append(eadc.shape[-1])

            dwi_file = case + '/rb1000.nii.gz'
            dwi = nib.load(dwi_file).get_fdata()

            label_file = case + '/rb1000_roi.nii.gz'
            label = nib.load(label_file).get_fdata()

            self.eadcs.append(eadc)
            self.dwis.append(dwi)
            self.labels.append(label)

    def __len__(self):
        return len(self.eadcs)

    def __getitem__(self, idx):
        # group = idx//self.num_gpus
        # max_len = max(self.all_len[group*self.num_gpus:group*self.num_gpus+self.num_gpus])
        max_len = max(self.all_len)
        
        eadc = self.eadcs[idx]
        this_len = eadc.shape[-1]
        eadc = eadc.astype(np.float32)
        eadc = np.transpose(eadc, (2, 0, 1))
        add_eadc = np.repeat(eadc[[-1]], max_len-this_len, axis=0)
        eadc = np.concatenate((eadc, add_eadc), axis=0)
        eadc = np.expand_dims(eadc, axis=1)

        dwi = self.dwis[idx]
        dwi = dwi.astype(np.float32)
        dwi = np.transpose(dwi, (2, 0, 1))
        add_dwi = np.repeat(dwi[[-1]], max_len-this_len, axis=0)
        dwi = np.concatenate((dwi, add_dwi), axis=0)        
        dwi = np.expand_dims(dwi, axis=1)
        
        input_data = np.concatenate((eadc, dwi), axis=1)

        label = self.labels[idx]
        label = label.astype(np.float32)
        label = np.transpose(label, (2, 0, 1))
        add_label = np.repeat(label[[-1]], max_len-this_len, axis=0)
        label = np.concatenate((label, add_label), axis=0)          
        label = np.expand_dims(label, axis=1)

        return input_data, label, this_len


class Stroke_lambda_test(Dataset):

    def __init__(self, data_dir, training_files, indx, seq_len):
        self.eadcs = []
        self.dwis = []
        self.labels = []
        self.seq_len = seq_len
        self.all_len = []
        self.all_cases = []
        training_cases = []
        train_file = open(training_files, "r")
        lines = train_file.readlines()
        for line in lines:
            if line != '\n':
                line = line.split('/')
                case = data_dir + line[1] + '/' + line[2]
                training_cases.append(case)

        all_files = glob.glob(os.path.join(data_dir, '*/*'))
        all_files.sort()
        val_list = list(set(all_files) - set(training_cases))
        val_list.sort()
        for case in [val_list[i] for i in indx]:
            eadc_file = case + '/eADC.nii.gz'
            eadc = nib.load(eadc_file).get_fdata()

            self.all_len.append(eadc.shape[-1])

            dwi_file = case + '/rb1000.nii.gz'
            dwi = nib.load(dwi_file).get_fdata()

            label_file = case + '/rb1000_roi.nii.gz'
            label = nib.load(label_file).get_fdata()

            case_name = case.split('/')
            case_name = case_name[-2] + '/' + case_name[-1]

            self.eadcs.append(eadc)
            self.dwis.append(dwi)
            self.labels.append(label)
            self.all_cases.append(case_name)

    def __len__(self):
        return len(self.eadcs)

    def __getitem__(self, idx):
        eadc = self.eadcs[idx]
        eadc = eadc.astype(np.float32)
        eadc = np.transpose(eadc, (2, 0, 1))
        eadc = np.expand_dims(eadc, axis=1)

        dwi = self.dwis[idx]
        dwi = dwi.astype(np.float32)
        dwi = np.transpose(dwi, (2, 0, 1)) 
        dwi = np.expand_dims(dwi, axis=1)
        
        input_data = np.concatenate((eadc, dwi), axis=1)

        label = self.labels[idx]
        label = label.astype(np.float32)
        label = np.transpose(label, (2, 0, 1))
        label = np.expand_dims(label, axis=1)

        case = self.all_cases[idx]

        return input_data, label, case

if __name__ == '__main__':
    np.random.seed(0) 
    indx = list(range(32))
    np.random.shuffle(indx)
    val_indx = indx[:20]
    test_indx = indx[20:]

    data_dir = '/home/yxo43/data/Stroke_AWS_DWI_Anom_PSU/'
    # data_dir = '/home/yxo43/data/stroke/Stroke_AWS_DWI_Anom_PSU_Partial_Truth/'
    training_cases = 'training_cases.txt'
    train_dataset = Stroke_lambda(data_dir, training_cases, 12)
    trainloader = data.DataLoader(train_dataset,
                batch_size=8, shuffle=True, num_workers=0)
    trainloader_iter = iter(trainloader)
    for epoch in range(200):
        data = next(trainloader_iter)

