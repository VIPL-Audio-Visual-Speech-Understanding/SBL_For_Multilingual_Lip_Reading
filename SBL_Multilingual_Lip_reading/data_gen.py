import pickle
import random
import cv2
import torch
import os
import math
import glob

from cvtransforms import *

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from config import IGNORE_ID, word_number, lrw_path, p, lrw_wav, mask, lrw1000_path, lrw1000_wav, lrw1000_info
from utils import extract_feature
from list_vocabs import words
#from list_vocabs_LRW1000 import LRW1000_words

import librosa
from g2p_en import G2p
g2p = G2p()

english_phonemes = {}
with open('English_phonemes.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
      items = line.rstrip('\n').strip('').split(' ')
      #print(items)
      english_phonemes[items[0]] = items[1]

chinese_phonemes={}

with open('chinese_phonemes_gai.txt', 'r') as file:
  lines = file.readlines()
  for line in lines:
    items = line.rstrip('\n').strip('').split('  ')
    #print(items)
    pinyin = items[0]
    phonemes = items[1].split(' ')
    chinese_phonemes[pinyin] = phonemes

total_phonemes = ['sos', 'eos', 's', 'p', 'ii', 'k', 'i', 'ng', 'l', 'e', 'v', 'e1', 'a1', 'm', 'z', 'zh', 'o', 'r', 'eu', 't', 'ai', 'h', 'th', 'y', 'n', 'ch', 'ae', 'au', 'er', 'd', 'f', 'ei', 'w', 'a', 'oi', 'b', 'uu', 'g', 'sh', 'dh', 'u', 'zh1', 'an', 'ang', 'en', 'eng', 'ie', 'in', 'ing', 'uo', 'ts', 'iii', 'ong', 'j', 'yu', 'yue', 'q', 'x']

w, h = 122, 122
th, tw = 112, 112

w_m, h_m = 96, 96
th_m, tw_m = 88, 88

x1_c = int(round((w - tw))/2.)
y1_c = int(round((h - th))/2.)
ll, tt = (224 - w) // 2, (224 - h) // 2

x1_c_m = int(round((w_m - tw_m))/2.)
y1_c_m = int(round((h_m - th_m))/2.)
ll_m, tt_m = 80, 116

def load_images(imgpath, st, ed, color_space='gray', is_training=False, max_len=30):
    imgs = []
    border_x = x1_c_m
    border_y = y1_c_m
    
    x1 = random.randint(0, border_x) if is_training else border_x
    y1 = random.randint(0, border_y) if is_training else border_y
    flip = is_training and random.random() > 0.5
    
    if ed > st + max_len: ed = st + max_len
    if st == ed: ed = st + 1 # safeguard
    files = [os.path.join(imgpath, '{:d}.jpg'.format(i)) for i in range(st, ed)]
    files = list(filter(lambda path: os.path.exists(path), files))
#     assert len(files) > 0, 'exception: {}, {}, {}'.format(imgpath, st, ed)
    
    for file in files:
        try:
            img = cv2.imread(file)
            rsz = cv2.resize(img, (96, 96))[y1: y1 + th_m, x1: x1 + tw_m]
            if flip: rsz = cv2.flip(rsz, 1)
            if color_space == 'gray': rsz = cv2.cvtColor(rsz, cv2.COLOR_BGR2GRAY)
            rsz = rsz / 255.
            if color_space == 'gray': rsz = ColorNormalize(rsz)
            imgs.append(rsz)
        except Exception as e:
            print (e, file)
    
    if color_space == 'gray':
        if len(imgs) == 0: seq = np.zeros((max_len, th_m, tw_m)) # safeguard
        else: seq = np.stack(imgs)[...] # gray: THW->CTHW
    else:
        if len(imgs) == 0: seq = np.zeros((3, max_len, th_m, tw_m))
        else: seq = np.stack(imgs).transpose(3, 0, 1, 2) # RGB: THWC->CTHW
    # pad to max number of timesteps
    if seq.shape[1] < max_len:
        to_pad = max_len - seq.shape[1]
        seq = np.pad(seq, ((0, 0), (0, to_pad), (0, 0), (0, 0)), 'constant')

    return seq.astype(np.float32)
    
def HorizontalFlip(batch_img):
    if random.random() > 0.5:
        batch_img = batch_img[:,:,::-1,...]
    return batch_img

def FrameRemoval(batch_img):
    for i in range(batch_img.shape[0]):
        if(random.random() < 0.05 and 0 < i):
            batch_img[i] = batch_img[i - 1]
    return batch_img

def FrameZero(batch_img):
    frameshape = batch_img[0].shape
    zero_img = np.zeros(frameshape)
    #zero_list = []
    for i in range(batch_img.shape[0]):
        if (random.random() < 0.1):
            batch_img[i] = zero_img
            #zero_list.append(i)
        else:
            batch_img[i] = batch_img[i]
    return batch_img
        
def load_file(filename):
    arrays = np.load(filename)
    arrays = arrays / 255.
    return arrays   

class AiShellDataset(Dataset):
    def __init__(self, args, split, kind):
        self.args = args
        self.split = split
        self.kind = kind
        self.samples = []
        max_length = 0
        npy_files = []
        npy_fold = glob.glob(os.path.join(lrw_path, '*'))
        
        if self.split == 'train':
            for fold in npy_fold:
                npy_files_fold = glob.glob(os.path.join(fold, split, '*.npy'))
                npy_files.extend(npy_files_fold[:int(len(npy_files_fold)*p)])

        for npy_file in npy_files:
            item = npy_file.split('/')[-1].split('_')[0]
            phonemes = g2p(item)
            trn = [total_phonemes.index(english_phonemes[one]) for one in phonemes]

            self.samples.append((npy_file, trn, 0))
        
        number_LRW = len(self.samples)
        
        if split == 'train':print('loading LRW {} {} samples...'.format(len(self.samples), split))
        
        if self.split == 'train':
            with open('trn1.txt', 'r') as file:
                lines = file.read().splitlines()
                lines = list(filter(lambda x:'7.31d3e1f43d431cecda814ff8ab3a4b437d' not in x, lines))        
                lines = list(filter(lambda x:x.strip(' ').split(',')[3] != 'C', lines))
                lines = list(filter(lambda x:x.strip(' ').split(',')[3] != 'n', lines))

            for line in lines:
                items = line.strip(' ').split(',')
                pinyins = items[3].split(' ')

                phonemes = []
                for pinyin in pinyins:
                    phones = chinese_phonemes[pinyin]
                    phonemes.extend(phones)
                trn = [total_phonemes.index(phoneme) for phoneme in phonemes]

                if len(trn) >= max_length:
                    max_length = len(trn)
                img_path = items[0]
                st,ed = int(float(items[4]) * 25) + 1, int(float(items[5]) * 25) + 1
                wav = os.path.join(lrw1000_wav, items[1]+'.wav')
                y, _ = librosa.load(wav, sr=16000)
                if len(y) > 0:
                    self.samples.append(((img_path,st,ed), trn, 1))
                    
        number_LRW1000 = (len(self.samples) - number_LRW)
        
        if split=='train':print('loading LRW1000 {} {} samples...'.format(number_LRW1000, split))
        
        if split=='train':print('loading LRW and LRW1000 {} {} samples...'.format(len(self.samples), split))

        #####val: loading val data#####
        if self.split == 'val' and self.kind == 'lrw':
            for fold in npy_fold:
                npy_files_fold = glob.glob(os.path.join(fold, split, '*.npy'))
                npy_files.extend(npy_files_fold[:int(len(npy_files_fold)*p)])

        for npy_file in npy_files:
            item = npy_file.split('/')[-1].split('_')[0]
            phonemes = g2p(item)
            trn = [total_phonemes.index(english_phonemes[one]) for one in phonemes]
            self.samples.append((npy_file, trn, 0))
            
        number_LRW_val = 0    
            
        if split=='val' and self.kind == 'lrw':
           print('loading LRW {} {} samples...'.format(len(self.samples), split))
           number_LRW_val = len(self.samples)

        if self.split == 'val' and self.kind == 'lrw1000':
            with open(os.path.join(lrw1000_info, 'val1.txt'), 'r') as file:
                lines = file.read().splitlines()
                lines = list(filter(lambda x:x.strip(' ').split(',')[3] != 'C', lines))
                lines = list(filter(lambda x:x.strip(' ').split(',')[3] != 'n', lines))

            for line in lines[:10000]:
                items = line.strip(' ').split(',')
                pinyins = items[3].split(' ')
                phonemes = []
                for pinyin in pinyins:
                    phones = chinese_phonemes[pinyin]
                    phonemes.extend(phones)
                trn = [total_phonemes.index(phoneme) for phoneme in phonemes]
                if len(trn) >= max_length:
                    max_length = len(trn)
                img_path = items[0]
                st,ed = int(float(items[4]) * 25) + 1, int(float(items[5]) * 25) + 1
                wav = os.path.join(lrw1000_wav, items[1]+'.wav')
                y, _ = librosa.load(wav, sr=16000)
                if len(y) > 0:
                    self.samples.append(((img_path,st,ed), trn, 1))

        #####test: loading test data#####
        if self.split == 'test' and self.kind == 'lrw':
            for fold in npy_fold:
                npy_files_fold = glob.glob(os.path.join(fold, split, '*.npy'))
                npy_files.extend(npy_files_fold[:int(len(npy_files_fold)*p)])

        for npy_file in npy_files:
            item = npy_file.split('/')[-1].split('_')[0]
            phonemes = g2p(item)
            trn = [total_phonemes.index(english_phonemes[one]) for one in phonemes]
            self.samples.append((npy_file, trn, 0))
            
        number_LRW_val = 0    
            
        if split=='test' and self.kind == 'lrw':
           print('loading LRW {} {} samples...'.format(len(self.samples), split))
           number_LRW_val = len(self.samples)

        if self.split == 'test' and self.kind == 'lrw1000':
            with open(os.path.join(lrw1000_info, 'tst1.txt'), 'r') as file:
                lines = file.read().splitlines()
                lines = list(filter(lambda x:x.strip(' ').split(',')[3] != 'C', lines))
                lines = list(filter(lambda x:x.strip(' ').split(',')[3] != 'n', lines))

            for line in lines:
                items = line.strip(' ').split(',')
                pinyins = items[3].split(' ')
                phonemes = []

                for pinyin in pinyins:
                    phones = chinese_phonemes[pinyin]
                    phonemes.extend(phones)
                trn = [total_phonemes.index(phoneme) for phoneme in phonemes]
                if len(trn) >= max_length:
                    max_length = len(trn)
                img_path = items[0]
                st,ed = int(float(items[4]) * 25) + 1, int(float(items[5]) * 25) + 1
                wav = os.path.join(lrw1000_wav, items[1]+'.wav')
                y, _ = librosa.load(wav, sr=16000)
                if len(y) > 0:
                    self.samples.append(((img_path,st,ed), trn, 1))

        print(max_length)
        
    def __getitem__(self, i):
        sample = self.samples[i]
        images = sample[0]
        trn = sample[1]
        indiction = sample[2]
        
        if indiction == 0:
            vid = load_file(images)
            vid = ColorNormalize(vid)

            if self.split=='train':
              vid = RandomCrop(vid, (88,88))
            else:
              vid = CenterCrop(vid, (88,88)) 

        if indiction == 1:
            imag_path,st,ed = images
            vid = load_images(os.path.join('../LRW1000', 'images', imag_path), st, ed, 'gray', self.split == 'train', max_len=30)
            
        if self.split == 'train':
            vid = HorizontalFlip(vid)
            vid = FrameRemoval(vid)
        
        length, w, h = vid.shape
        vids = np.zeros((30, w, h), dtype=np.float32)
        
        vids[:length, :, :] = vid
        trn_reverse = []
        for i in trn[::-1]:
            trn_reverse.append(i)

        labels = np.pad(trn, (0, 14 - len(trn)), 'constant', constant_values=IGNORE_ID)
        labels_reverse = np.pad(trn_reverse, (0, 14-len(trn)), 'constant', constant_values=IGNORE_ID)

        return torch.FloatTensor(vids), torch.LongTensor(labels), torch.LongTensor(labels_reverse), indiction

    def __len__(self):
        return len(self.samples)

    def _load_vid(self, p):
        files = sorted(os.listdir(p), key=lambda x:int(x.split('.')[0]))        
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        array = [cv2.resize(im, (128, 128)) for im in array]
        array = np.stack(array, axis=0)
        return array

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.vstack(array)