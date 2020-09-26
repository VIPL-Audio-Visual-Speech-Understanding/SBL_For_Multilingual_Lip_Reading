import os
import pickle

from tqdm import tqdm

from config import pickle_file, tran_file,  lrw_path, p
from utils import ensure_folder

import glob

from g2p_en import G2p
g2p = G2p()
# letters = ['<sos>', '<eos>', ' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 
# 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
phonemes_dict = {}
with open('English_phonemes.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
      items = line.rstrip('\n').strip('').split(' ')
      print(items)
      phonemes_dict[items[0]] = items[1]
      

def get_data(split):
    print('getting {} data...'.format(split))

    global VOCAB
    samples = []
    
    npy_files = []
    npy_fold = glob.glob(os.path.join(lrw_path, '*'))
    if split == 'train':
            for fold in npy_fold:
                npy_files_fold = glob.glob(os.path.join(fold, split, '*.npy'))
                npy_files.extend(npy_files_fold[:int(len(npy_files_fold)*p)])
    else:
            npy_files = glob.glob(os.path.join(lrw_path, '*', split, '*.npy'))

    for npy_file in npy_files:
            item = npy_file.split('/')[-1].split('_')[0]
            phonemes = g2p(item)
            #print(npy_file, item, phonemes)
            
            for one in phonemes:
                build_vocab(phonemes_dict[one])
            trn = [VOCAB[phonemes_dict[phoneme]] for phoneme in phonemes]
            #trn = np.pad(trn, (0, max_target_lengths - len(trn)), 'constant', constant_values=IGNORE_ID)
            
            samples.append({'name':npy_file, 'trn':trn, 'item':item})
        
    print(len(samples))
    print('loading {} {} samples...'.format(len(samples), split)) 
    
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
    VOCAB = {'<sos>': 0, '<eos>': 1}
    IVOCAB = {0: '<sos>', 1: '<eos>'}
    data = dict()
    data['VOCAB'] = VOCAB
    data['IVOCAB'] = IVOCAB
    data['train'] = get_data('train')
    data['val'] = get_data('val')
    data['tst'] = get_data('tst')


    with open(pickle_file, 'wb') as file:
        pickle.dump(data, file)

    print('num_train: ' + str(len(data['train'])))
    print('num_val: ' + str(len(data['val'])))
    print('num_tst: ' + str(len(data['tst'])))