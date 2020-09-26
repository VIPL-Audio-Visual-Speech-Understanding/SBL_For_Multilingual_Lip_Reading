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

from config import pickle_file, IGNORE_ID, word_number, lrw_path, p, lrw_wav, mask
from utils import extract_feature
from list_vocabs import words

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
    
def ColorNormalize(batch_img):
    batch_img = batch_img / 255.0
    return batch_img

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

class AiShellDatasetLRW(Dataset):
    def __init__(self, args, split):
        self.args = args
        self.split = split
        self.samples = []
        npy_files = []
        npy_fold = glob.glob(os.path.join(lrw_path, '*'))
        if self.split == 'train':
            for fold in npy_fold:
                npy_files_fold = glob.glob(os.path.join(fold, split, '*.npy'))
                npy_files.extend(npy_files_fold[:int(len(npy_files_fold)*p)])
        else:
            npy_files = glob.glob(os.path.join(lrw_path, '*', split, '*.npy'))

        for npy_file in npy_files:
            item = npy_file.split('/')[-1].split('_')[0]
            wav_file = npy_file.replace('/train/roi_80_116_175_211_npy_gray', lrw_wav).replace('npy', 'wav')
            number = int(npy_file.split('/')[-1].split('_')[1].split('.')[0])
            label = words.index(item)
            self.samples.append((npy_file, wav_file, label, 0))
        
        print(len(self.samples))
        print('loading {} {} samples...'.format(len(self.samples), split))

    def __getitem__(self, i):
        sample = self.samples[i]
        images = sample[0]
        wav = sample[1]
        trn = sample[2]
        indiction = sample[3]
        vid = load_file(images)

        feature = extract_feature(input_file=wav, feature='fbank', dim=self.args.d_input, cmvn=True)
        aud = build_LFR_features(feature, m=self.args.LFR_m, n=self.args.LFR_n)
        
        if self.split == 'train':
            vid = HorizontalFlip(vid)
            vid = FrameRemoval(vid)
        
        length, w, h = vid.shape
        vids = np.zeros((31, w, h), dtype=np.float32)
        auds = np.zeros((42, 320), dtype=np.float32)
        
        vids[:length, :, :] = vid
        auds[:len(aud), :] = aud
        
        ####random choose 14 sequences to 0
        '''
        indices = [random.randint(0,29) for _ in range(int(29*mask))]
        zeros_seq = np.zeros((w, h), dtype=np.float32)
        for indice in indices:
            vids[indice] = zeros_seq
        '''    
        #vid = vid.copy()
        # print(vid.shape)
        # print(aud.shape)

        # auds = np.zeros((41, 320), dtype=np.float32)
        # aud = self._padding(aud, 41)
        # auds[:len(aud), :] = aud

        return vids, torch.FloatTensor(auds), trn, indiction
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,pin_memory=True, shuffle=True, num_workers=args.num_workers)

    valid_dataset = AiShellDataset(args, 'val')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers)

    tst_dataset = AiShellDataset(args, 'test')
    tst_loader = torch.utils.data.DataLoader(tst_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False, num_workers=args.num_workers)

    print('train_dataset: ', len(train_dataset), len(train_loader))
    print('valid_dataset: ', len(valid_dataset), len(valid_loader))
    print('tst_dataset: ', len(tst_dataset), len(tst_loader))
