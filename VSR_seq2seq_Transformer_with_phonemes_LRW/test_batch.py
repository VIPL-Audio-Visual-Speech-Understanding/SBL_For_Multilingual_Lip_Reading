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
    parser.add_argument('--decode_max_len', default=100, type=int,
                        help='Max output length. If ==0 (default), it uses a '
                             'end-detect function to automatically find maximum '
                             'hypothesis lengths')
    args = parser.parse_args()
    return args

#from utils import parse_args
args = parse_args()
valid_dataset = AiShellDataset(args, 'val')
valid_loader = DataLoader(valid_dataset, batch_size=180, pin_memory=True, shuffle=True, num_workers=16)

def wer_compute(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        #print(word_pairs)
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()

def per_compute(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return np.array(cer).mean()  

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
    num_samples = len(samples)
    #num_samples = 500
    total_cer = 0
    
    predict_txt = []
    truth_txt = []

    pred_all_txt = []
    gold_all_txt = []
    
    pred_phonemes = []
    gold_phonemes = []
    
    for data in tqdm(valid_loader):
        padded_input, padded_target = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.numpy()
        
        #print(padded_input.size())
        input = padded_input.unsqueeze(-1)
        #input = input.unsqueeze(-1)
        #print(input.size())
        with torch.no_grad():
            nbest_hyps = model.recognize(input, char_list, args)
        
        #print(padded_target, len(padded_target))
        #print(nbest_hyps, len(nbest_hyps))
        
        for n in range(len(nbest_hyps)):
                #changdu = len(gold[n].cpu().numpy())
                golds = [char_list[one] for one in padded_target[n] if one not in (sos_id, eos_id, -1)]
                changdu = len(golds)
                preds = [char_list[one] for one in nbest_hyps[n][:(changdu+1)] if one not in (sos_id, eos_id, -1)]
                    
                pred_all_txt.append(''.join(preds))
                gold_all_txt.append(''.join(golds))
                
                cer = cer_function(golds, preds)
                total_cer += cer
       
    avg_cer = total_cer / num_samples
    avg_wer = wer_function(pred_all_txt, gold_all_txt)
    print('avg_cer: ' + str(avg_cer))
    print('avg_wer: ' + str(avg_wer))
