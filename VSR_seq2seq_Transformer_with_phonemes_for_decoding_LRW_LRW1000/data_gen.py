import pickle
import random
import cv2
import torch
import os
import math
import glob

import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

from config import pickle_file, IGNORE_ID, word_number, lrw1000_path, p, lrw1000_wav, mask, lrw1000_info
from utils import extract_feature
from list_vocabs import words

import librosa
from cvtransforms import *

lrw1000_phonemes = ['sos', 'eos', 's', 'au', 'm', 'i', 'p', 'ii', 't', 'q', 'yu', 'x', 'j', 'an', 'y', 'eu', 'sh', 'iii', 'd', 'ong', 'ang', 'zh', 'l', 'e1', 'f', 'g', 'eng', 'ts', 'uo', 'a', 'ch', 'w', 'en', 'h', 'u', 'ai', 'yue', 'uu', 'in', 'ing', 'ei', 'z', 'b', 'zh1', 'k', 'ie', 'er', 'n']

chinese_phonemes={}

with open('chinese_phonemes_gai.txt', 'r') as file:
  lines = file.readlines()
  for line in lines:
    items = line.rstrip('\n').strip('').split('  ')
    #print(items)
    pinyin = items[0]
    phonemes = items[1].split(' ')
    chinese_phonemes[pinyin] = phonemes

#print(chinese_phonemes)

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
        if len(imgs) == 0: seq = np.zeros((1, max_len, th_m, tw_m)) # safeguard
        else: seq = np.stack(imgs)[None, ...] # gray: THW->CTHW
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
    # arrays = np.stack([cv2.cvtColor(arrays[_], cv2.COLOR_BGR2GRAY)
    #                   for _ in range(29)], axis=0)
    arrays = arrays / 255.
    return arrays   

def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')

    for elem in batch:
        visual, feature, trn = elem
        max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
        #max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

    for i, elem in enumerate(batch):
        visual, feature, trn = elem
        input_length = feature.shape[0]
        input_dim = feature.shape[1]
        padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
        padded_input[:input_length, :] = feature
        #padded_target = np.pad(trn, (0, max_target_len - len(trn)), 'constant', constant_values=IGNORE_ID)
        batch[i] = (visual, padded_input, trn)

    # sort it by input lengths (long to short)
    #batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)

def build_LFR_features(inputs, m, n):
    LFR_inputs = []
    T = inputs.shape[0]
    T_lfr = int(np.ceil(T / n))
    for i in range(T_lfr):
        if m <= T - i * n:
            LFR_inputs.append(np.hstack(inputs[i * n:i * n + m]))
        else:  # process last LFR frame
            num_padding = m - (T - i * n)
            frame = np.hstack(inputs[i * n:])
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            LFR_inputs.append(frame)

    return np.vstack(LFR_inputs)

class AiShellDataset(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.samples = []
        durations = []
        max_length = 0
        #-----------------------------------------------------------
        if self.split == 'train':
            with open('trn_1000_{}.txt'.format(p), 'r') as file:
                lines = file.read().splitlines()
                lines = list(filter(lambda x:'7.31d3e1f43d431cecda814ff8ab3a4b437d' not in x, lines))
                lines = list(filter(lambda x:x.strip(' ').split(',')[3] != 'C', lines))
                lines = list(filter(lambda x:x.strip(' ').split(',')[3] != 'n', lines))
            #print(len(lines))
            for line in lines:
                items = line.strip(' ').split(',')
                pinyins = items[3].split(' ')
                #print(pinyins)
                phonemes = []
                for pinyin in pinyins:
                    phones = chinese_phonemes[pinyin]
                    phonemes.extend(phones)
                trn = [lrw1000_phonemes.index(phoneme) for phoneme in phonemes]
                if len(trn) >= max_length:
                    max_length = len(trn)
                #print(phonemes)
                #print(trn)
                #label = words.index(items[3])
                #print(items[3])
                img_path = items[0]
                st,ed = int(float(items[4]) * 25) + 1, int(float(items[5]) * 25) + 1
                #duration = ed - st
                #durations.append(duration)
                wav = os.path.join(lrw1000_wav, items[1]+'.wav')
                y, _ = librosa.load(wav, sr=16000)
                if len(y) > 0:
                    self.samples.append(((img_path,st,ed), trn))
        
        elif self.split == 'val':
            with open(os.path.join(lrw1000_info, 'val1.txt'), 'r') as file:
                lines = file.read().splitlines()
                lines = list(filter(lambda x:x.strip(' ').split(',')[3] != 'C', lines))
                lines = list(filter(lambda x:x.strip(' ').split(',')[3] != 'n', lines))
            print(len(lines))
            for line in lines:
                items = line.strip(' ').split(',')
                pinyins = items[3].split(' ')
                #print(pinyins)
                phonemes = []
                for pinyin in pinyins:
                    phones = chinese_phonemes[pinyin]
                    phonemes.extend(phones)
                trn = [lrw1000_phonemes.index(phoneme) for phoneme in phonemes]
                #print(phonemes)
                #print(trn)
                
                img_path = items[0]
                st,ed = int(float(items[4]) * 25) + 1, int(float(items[5]) * 25) + 1
                #duration = ed - st
                #durations.append(duration)
                wav = os.path.join(lrw1000_wav, items[1]+'.wav')
                y, _ = librosa.load(wav, sr=16000)
                if len(y) > 0:
                    self.samples.append(((img_path,st,ed), trn))
        print('max length is: ', max_length)
        print(len(self.samples))
        print('loading {} {} samples...'.format(len(self.samples), split))
    
    ###max frames=57
    def __getitem__(self, i):
        sample = self.samples[i]
        
        imag_path,st,ed = sample[0]
        
        #wav = sample[1]
        trn = sample[1]
        #print(trn)
        trn1 = []
        for i in trn[::-1]:
            trn1.append(i)
        
        #print(trn)
        #print(trn1)
        labels_reverse = np.pad(trn1, (0, 14 - len(trn1)), 'constant', constant_values=IGNORE_ID)
        labels = np.pad(trn, (0, 14 - len(trn)), 'constant', constant_values=IGNORE_ID)
        #vid = load_file(images)
        vid_img = load_images(os.path.join('/data/lip/LRW1000', 'images', imag_path), st, ed, 'gray', self.split == 'train', max_len=30)
        vid = vid_img
        
        if self.split == 'train':
            vid = HorizontalFlip(vid)
            vid = FrameRemoval(vid)
        #print(vid.shape)
        vid = vid.squeeze(0)

        return torch.FloatTensor(np.ascontiguousarray(vid)), torch.LongTensor(labels), torch.LongTensor(labels_reverse)
        #return vid, trn

    def __len__(self):
        return len(self.samples)

    def _load_vid(self, p): 
        #files = sorted(os.listdir(p))
        files = sorted(os.listdir(p), key=lambda x:int(x.split('.')[0]))        
        array = [cv2.imread(os.path.join(p, file)) for file in files]
        array = list(filter(lambda im: not im is None, array))
        # array = [cv2.resize(im, (50, 100)).reshape(50, 100, 3) for im in array]
        array = [cv2.resize(im, (128, 128)) for im in array]
        #print(p, len(array))
        array = np.stack(array, axis=0)
        return array

    # def _load_boundary(self, arr, time):
    #     duration = math.ceil(float(time)*25)
    #     st = math.ceil(((30-duration)/2))
    #     ed = math.floor(((30+duration)/2))
    #     #print(st, ed)
    #     return arr[st:ed]

    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.vstack(array)


if __name__ == "__main__":
    import torch
    from utils import parse_args
    from tqdm import tqdm

    args = parse_args()
    #train_dataset = AiShellDataset(args, 'train')

    train_dataset = AiShellDataset(args, 'train')
    #print(train_dataset[0][0].size())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=pad_collate, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    valid_dataset = AiShellDataset(args, 'val')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, collate_fn=pad_collate, pin_memory=True, shuffle=False, num_workers=args.num_workers)

    print('train_dataset: ', len(train_dataset), len(train_loader))
    print('valid_dataset: ', len(valid_dataset), len(valid_loader))