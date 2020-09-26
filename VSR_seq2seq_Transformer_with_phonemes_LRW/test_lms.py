import numpy as np
import torch
import os
import torch.nn as nn
from tensorboardX import SummaryWriter
# from torch import nn
from tqdm import tqdm
import editdistance

from config import device, print_freq, vocab_size, sos_id, eos_id, IGNORE_ID, word_length
from data_gen import AiShellDataset, pad_collate
from transformer.decoder import Decoder
from transformer.encoder import Encoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence

import pickle
pklfile = 'char_list.pkl'
with open(pklfile, 'rb') as file:
    data = pickle.load(file)
char_list = data
print(char_list)
#torch.cuda.set_device(0)
#os.environ['CUDA_VISIBLE_DEVICES']='2,3,0'

def wer_compute(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        #print(word_pairs)
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()

def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    #writer = SummaryWriter()
    epochs_since_improvement = 0

    # Initialize / load checkpoint
    if checkpoint is None:
        # model
        encoder = Encoder(512, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        decoder = Decoder(sos_id, eos_id, vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)
        model = Transformer(encoder, decoder)
        # print(model)
        #model = nn.DataParallel(model, device_ids=[2,3]).cuda()

        # optimizer
        optimizer = TransformerOptimizer(
            torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    else:
        checkpoint = 'BEST_checkpoint_LRW_phonemes_decoding.tar'
        checkpoint = torch.load(checkpoint)
        #start_epoch = checkpoint['epoch'] + 1
       # epochs_since_improvement = checkpoint['epochs_since_improvement']
        model = checkpoint['model']
        print('loading model successful....')
       # optimizer = checkpoint['optimizer']
      #  optimizer._update_lr()
        #lr = 0.0002
        # optimizer = TransformerOptimizer(torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)
    #model = model.module.module.module.module.module.module
    #model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])

    # Custom dataloaders
    valid_dataset = AiShellDataset(args, 'val')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=20, collate_fn=pad_collate, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    # Epochs
    k = 0
    for epoch in range(start_epoch, args.epochs):
        if epoch == 1:
            break
        
        valid_loss, wer = valid(valid_loader=valid_loader,model=model,logger=logger)

def valid(valid_loader, model, logger):
    model.eval()

    losses = AverageMeter()
    pred_all_txt = []
    gold_all_txt = []
    # Batches
    wer = float(0)
    print('wer: ', wer)
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        #print(data[1])
        padded_input, padded_target, _ = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        #input_lengths = input_lengths.to(device)
        #print(padded_input.size())
        #print(padded_target)
        if padded_target.size(1) <= word_length:
            with torch.no_grad():
                # Forward prop.
                pred, gold = model(padded_input, padded_target)
                loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
                print(pred)
                pred_argmax = pred.argmax(-1)
                preds = pred_argmax
                #print(preds, preds.cpu().numpy())
                pred_txt = []
                for arr in preds.cpu().numpy():
                    # preds = [char_list[one] for one in arr if one not in (sos_id, eos_id) and ]
                    preds = []
                    for one in arr:
                        if one != eos_id:
                            preds.append(char_list[one])
                        else:
                            break
                    pred_txt.append(' '.join(preds))
                    #print(pred_txt)
                pred_all_txt.extend(pred_txt)
                #print('predict txt: ', pred_all_txt)

                gold_txt = []
                for arr in gold.cpu().numpy():
                    #print(arr)
                    #golds = []
                    # for one in arr:
                    #     if one == -1 or one ==1:
                    #         golds.append('')
                    #     else:
                    #         golds.append(char_list[one])

                    golds = [char_list[one] for one in arr if one not in (sos_id, eos_id, -1)]
                    gold_txt.append(' '.join(golds))
                    #print(gold_txt)

                gold_all_txt.extend(gold_txt)
                for preds,gt in zip(pred_txt, gold_txt):
                    print('preds: ', preds)
                    print('gt: ', gt)
                #print('ground truth txt: ', gold_all_txt)
                #print(' '.join(golds))
                #print(pred_argmax, gold)

            # Keep track of metrics
            losses.update(loss.item())
            
        else:
            break
        # print('\n')
        # print('pred_results: ',pred_all_txt[:10])
        # print('gold_results: ',gold_all_txt[:10])
        # print('\n')
        #print(pred_all_txt)
    #print(gold_all_txt)

    wer = wer_compute(pred_all_txt, gold_all_txt)
    print('wer: ', wer)
    
    # Print status
    logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    return losses.avg, wer


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
