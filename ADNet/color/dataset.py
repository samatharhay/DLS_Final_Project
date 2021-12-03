import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
import torch
import torch.nn as nn
import scipy.io
from skimage.transform import resize
from utils import data_augmentation

def normalize(data):
    return data/255.

def add_img_to_dataset(h5f, img, name):
    img = img.transpose(2,0,1)
    img = np.float32(normalize(img))
    h5f.create_dataset(name, data=img)

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    #print endc
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(opt, data_path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    files = glob.glob(os.path.join(data_path, 'SIDD-med', '*'))
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        h, w, c = img.shape
        if opt.aug_train_data:
            scales = [1.0, 0.9, 0.8, 0.7]
            for scale in scales:
                rescaled = cv2.resize(img, (int(h*scale), int(w*scale)), interpolation=cv2.INTER_CUBIC)
                rescaled = rescaled.transpose(2,0,1)
                rescaled = np.float32(normalize(rescaled))
                patches = Im2Patch(rescaled, win=patch_size, stride=stride)
                for n in range(patches.shape[3]):
                    data = patches[:,:,:,n].copy()
                    h5f.create_dataset(str(train_num), data=data)
                    train_num += 1
                    for m in range(aug_times-1):
                        data_aug = data_augmentation(data, np.random.randint(1,8))
                        h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                        train_num += 1
        else:
            add_img_to_dataset(h5f, img, str(train_num))
            train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    h5f = h5py.File('val.h5', 'w')
    if opt.test_data == "SIDD":
        benchmark = np.array(scipy.io.loadmat('data/ValidationGtBlocksSrgb.mat')['ValidationGtBlocksSrgb'])
        benchmark = benchmark.reshape(-1, 256, 256, 3)
        for val_num, img in enumerate(benchmark):
            img = resize(img[:], (50, 50))
            add_img_to_dataset(h5f, img, str(val_num))
    else:
        files = glob.glob(os.path.join(data_path, opt.test_data, '*'))
        files.sort()
        for val_num, f in enumerate(files):
            img = cv2.imread(f)
            add_img_to_dataset(h5f, img, str(val_num))
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % (val_num+1))

class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
