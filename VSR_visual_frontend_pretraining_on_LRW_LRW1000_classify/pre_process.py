import os
import pickle

from tqdm import tqdm

from config import pickle_file, lrw_path, lrw_info, lrw_wav
from utils import ensure_folder

import glob
import numpy as np

letters = ['<sos>', '<eos>', ' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 
'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def get_data(split):
    print('getting {} data...'.format(split))

    global VOCAB

    npy_files = glob.glob(os.path.join(lrw_path, '*', split, '*.npy'))
    #print(wav_files)
    print(npy_files[:10])
    
    samples = []
    for npy_file in npy_files:
        items = npy_file.split(os.path.sep)
        text = items[-1][:-10]
        #print(text)
        images = npy_file
        if os.path.exists(images) :
            info = npy_file[:-4].replace('roi_80_116_175_211_npy_gray', 'LRW_TXT')+'.txt'
            with open(info, 'r') as f:
                time = f.readlines()[-1].rstrip('\n').strip(' ').split(' ')[1]
            #trn = [letters.index('<sos>')]
            #trn = list(txt) + ['<eos>']
            #print(trn)
            wav_file = npy_file[:-4].replace('roi_80_116_175_211_npy_gray', 'lrw_wav/lrw_mp4')+'.wav'
            trn = list(text)
            #print(trn)
            for c in (trn):
                build_vocab(c)
            trn = [VOCAB[c] for c in trn]
        #print(trn)
        # for i in range(35 - len(trn)):
        #     trn.append(1)
        
        #print({'trn':trn, 'wave':wav_file, 'images':images})
            samples.append({'trn':trn, 'wave':wav_file, 'images':images, 'time':time})
        # print(trn)
        # print(text)
        # print(items)
    print('split: {}, num_files: {}'.format(split, len(samples)))
    #print(samples)
    return samples

def build_vocab(token):
    global VOCAB, IVOCAB
    if not token in VOCAB:
        next_index = len(VOCAB)
        VOCAB[token] = next_index
        IVOCAB[next_index] = token
    # with open(tran_file, 'r', encoding='utf-8') as file:
    #     lines = file.readlines()

if __name__ == "__main__":
    VOCAB = {'<sos>': 0, '<eos>': 1, 'Z':27}
    IVOCAB = {0: '<sos>', 1: '<eos>', 27: 'Z'}
    data = dict()
    data['VOCAB'] = VOCAB
    data['IVOCAB'] = IVOCAB
    data['train'] = get_data('train')
    data['val'] = get_data('val')
    data['test'] = get_data('test')


    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_val: ' + str(len(data['val'])))
    print('num_test: ' + str(len(data['test'])))