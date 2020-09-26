import argparse
import pickle
import os,math
import cv2
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from config import pickle_file, device, input_dim, LFR_m, LFR_n, sos_id, eos_id, word_length
from data_gen import AiShellDataset
from utils import extract_feature
from xer import cer_function, wer_function
from torch.utils.data import Dataset, DataLoader

def HorizontalFlip(batch_img):
    if random.random() > 0.5:
        batch_img = batch_img[:,:,::-1,...]
    return batch_img

def FrameRemoval(batch_img):
    for i in range(batch_img.shape[0]):
        if(random.random() < 0.05 and 0 < i):
            batch_img[i] = batch_img[i - 1]
    return batch_img
    
    
def ColorNormalize(batch_img):
    batch_img = batch_img / 255.0
    return batch_img

def parse_args():
    parser = argparse.ArgumentParser(
        "End-to-End Automatic Speech Recognition Decoding.")
    # decode
    parser.add_argument('--beam_size', default=1, type=int,
                        help='Beam size')
    parser.add_argument('--nbest', default=1, type=int,
                        help='Nbest size')
    parser.add_argument('--decode_max_len', default=0, type=int,
                        help='Max output length. If ==0 (default), it uses a '
                             'end-detect function to automatically find maximum '
                             'hypothesis lengths')
    args = parser.parse_args()
    return args

#from utils import parse_args
args = parse_args()
valid_dataset = AiShellDataset(args, 'val')
valid_loader = DataLoader(valid_dataset, batch_size=10, pin_memory=True, shuffle=True, num_workers=16)

if __name__ == '__main__':
    args = parse_args()
    samples = valid_dataset
    with open(pickle_file, 'rb') as file:
        data = pickle.load(file)
    char_list = data['IVOCAB']

    checkpoint = 'BEST_checkpoint_LRW_phonemes_decoding.tar'
    checkpoint = torch.load(checkpoint, map_location='cpu')
    model = checkpoint['model'].to(device)
    model = model.module
    #print(model)
    #if isinstance(model, torch.nn.DataParallel):
     #   model = model.module.module.module
    #model = nn.DataParallel(model, device_ids=[0])
    model.eval()
    #print(model)
    samples = samples
    #num_samples = len(samples)
    num_samples = 1
    total_cer = 0
    
    predict_txt = []
    truth_txt = []

    for i in tqdm(range(num_samples)):
        sample = samples[i]
        # wave = sample['wave']
        # trn = sample['trn']
        # images = sample['images']
        # feature = extract_feature(input_file=wave, feature='fbank', dim=input_dim, cmvn=True)
        # feature = build_LFR_features(feature, m=LFR_m, n=LFR_n)
        # feature = np.expand_dims(feature, axis=0)
        input = sample[0].to(device)
        trn = sample[1]
        trn = trn.numpy()
        #input_length = [input[0].shape[0]]
        #input_length = torch.LongTensor(input_length).to(device)
        #print(input.size())
        input = input.unsqueeze(0)
        input = input.unsqueeze(-1)
        #print(input.size())
        with torch.no_grad():
            nbest_hyps = model.recognize(input, char_list, args)
        
        #print(trn)
        gt = [char_list[idx] for idx in trn if idx not in (sos_id, eos_id, -1)]
        gt_list = gt
        length = len(gt)
        gt = ''.join(gt)
        truth_txt.append(gt) 

        #print(gt_list)
        
        hyp_list = []
        #print(nbest_hyps)
        #for hyp in nbest_hyps:
        out = nbest_hyps[0]['yseq']
        #print(out)
        out = [char_list[idx] for idx in out if idx not in (sos_id, eos_id, -1)]
        hyp_list = out[:length]
        out = ''.join(out[:length])
        #hyp_list.append(out)
        predict_txt.append(out)

        #print(hyp_list)

        cer = cer_function(gt_list, hyp_list)
        total_cer += cer
       
    avg_cer = total_cer / num_samples
    avg_wer = wer_function(predict_txt, truth_txt)
    print('avg_cer: ' + str(avg_cer))
    print('avg_wer: ' + str(avg_wer))
