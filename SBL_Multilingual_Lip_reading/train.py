import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch import nn
from tqdm import tqdm
import editdistance
from cvtransforms import *

from config import device, print_freq, sos_id, eos_id, word_number, p, vocab_size
from data_gen import AiShellDataset
#from data_gen_LRW import AiShellDatasetLRW

from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

import pickle

word_length = 14
#from list_vocabs import words
total_phonemes = ['sos', 'eos', 's', 'p', 'ii', 'k', 'i', 'ng', 'l', 'e', 'v', 'e1', 'a1', 'm', 'z', 'zh', 'o', 'r', 'eu', 't', 'ai', 'h', 'th', 'y', 'n', 'ch', 'ae', 'au', 'er', 'd', 'f', 'ei', 'w', 'a', 'oi', 'b', 'uu', 'g', 'sh', 'dh', 'u', 'zh1', 'an', 'ang', 'en', 'eng', 'ie', 'in', 'ing', 'uo', 'ts', 'iii', 'ong', 'j', 'yu', 'yue', 'q', 'x']

def wer_compute(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        #print(word_pairs)
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()

def wer_compute(predict, truth):        
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        #print(word_pairs)
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return np.array(wer).mean()

def per_compute(predict, truth):        
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return np.array(cer).mean()

def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    epochs_since_improvement = 0
    #pt = 'acc0.84412.pt'
    #pt='acc0.38423358796386053.pt'
    pt=None
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
        
        model = Transformer(encoder, decoder, pt)
        # print(model)
        #model = nn.DataParallel(model, device_ids=[0,1,2])

        # optimizer
        optimizer = TransformerOptimizer(
            torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    else:
        encoder = Encoder(512, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
        
        decoder = Decoder(sos_id, eos_id, vocab_size,
                          args.d_word_vec, args.n_layers_dec, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout,
                          tgt_emb_prj_weight_sharing=args.tgt_emb_prj_weight_sharing,
                          pe_maxlen=args.pe_maxlen)
        
        model = Transformer(encoder, decoder, pt)
        
        checkpoint_name = checkpoint
        checkpoint = torch.load(checkpoint)
        print('loading model parameters successful--->{}!'.format(checkpoint_name))
        pretrain_model = checkpoint['model']
        pretrained_dict = pretrain_model.module.state_dict()
        
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}  ###and ('encoder_v' not in k)}  ###
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        print('miss matched params:{}'.format(missed_params))              
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)                             
        
        #optimizer = checkpoint['optimizer']
        #optimizer = TransformerOptimizer(torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))
        optimizer = TransformerOptimizer(torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

        optimizer._update_lr()

    logger = get_logger()

    # Move to GPU, if available
    model = model.to(device)
    model = nn.DataParallel(model, device_ids=[0,1,2,3,4,5,6,7])
    # define a criterion 
    criterion = nn.CrossEntropyLoss()
    criterion_KD = nn.KLDivLoss()

    # Custom dataloaders

    train_dataset = AiShellDataset(args, 'train', 'all')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
    
    valid_dataset_LRW = AiShellDataset(args, 'val', 'lrw')
    valid_loader_LRW = torch.utils.data.DataLoader(valid_dataset_LRW, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    valid_dataset_LRW1000 = AiShellDataset(args, 'val', 'lrw1000')
    valid_loader_LRW1000 = torch.utils.data.DataLoader(valid_dataset_LRW1000, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
    '''
    train_dataset = AiShellDataset(args, 'test', 'all')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
    
    valid_dataset_LRW = AiShellDataset(args, 'test', 'lrw')
    valid_loader_LRW = torch.utils.data.DataLoader(valid_dataset_LRW, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    valid_dataset_LRW1000 = AiShellDataset(args, 'test', 'lrw1000')
    valid_loader_LRW1000 = torch.utils.data.DataLoader(valid_dataset_LRW1000, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
    '''
    # Epochs
    k = 0
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        
        train_loss, n = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger, k=k)
        k = n
        print('train_loss: ', train_loss)

        lr = optimizer.lr
        print('\nLearning rate: {}'.format(lr))

        step_num = optimizer.step_num
        print('Step num: {}\n'.format(step_num))
        
        # One epoch's validation
        l2r_wer_lrw, l2r_per_lrw, r2l_wer_lrw, r2l_per_lrw = valid_lrw(valid_loader=valid_loader_LRW,model=model,logger=logger)
        l2r_wer_lrw1000, l2r_per_lrw1000, r2l_wer_lrw1000, r2l_per_lrw1000 = valid_lrw1000(valid_loader=valid_loader_LRW1000,model=model,logger=logger)
        
        # Check if there was an improvement
        is_best = (l2r_wer_lrw + l2r_wer_lrw1000) < best_loss

        best_loss = min((l2r_wer_lrw + l2r_wer_lrw1000), best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)

def train(train_loader, model, optimizer, epoch, logger, k):
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    # Batches
    n = k
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        padded_input, padded_target, padded_target_r2l, _ = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        pred_l2r, gold_l2r, pred_r2l, gold_r2l = model(padded_input, padded_target, padded_target_r2l)
        
        loss_l2r, n_correct_l2r = cal_performance(pred_l2r, gold_l2r, smoothing=args.label_smoothing)
        loss_r2l, n_correct_r2l = cal_performance(pred_r2l, gold_r2l, smoothing=args.label_smoothing)

        loss = 0.5*(loss_l2r+loss_r2l)
        #loss = loss_r2l
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        n += 1
        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % 1 == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(train_loader), loss=losses))

    return losses.avg, n

def valid_lrw(valid_loader, model, logger):
    model = model.module
    model.eval()

    pred_all_txt = []
    gold_all_txt = []
    
    pred_phonemes = []
    gold_phonemes = []
    
    pred_all_txt_r2l = []
    gold_all_txt_r2l = []
    
    pred_phonemes_r2l = []
    gold_phonemes_r2l = []
    # Batches
    wer = float(0)
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input, padded_target, padded_target_reverse, _ = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        padded_target_reverse = padded_target_reverse.to(device)

        with torch.no_grad():
                # Forward prop.
                gold_l2r = padded_target
                gold_r2l = padded_target_reverse
                pred_l2r, pred_r2l = model.recognize(padded_input)

                predss_l2r = pred_l2r
                predss_r2l = pred_r2l

                pred_txt = []
                gold_txt = []
                pred_txt_r2l = []
                gold_txt_r2l = []
                length = predss_l2r.size(0)
                length_r2l = predss_r2l.size(0)
                for n in range(length):
                    golds = [total_phonemes[one] for one in gold_l2r[n].cpu().numpy() if one not in (sos_id, eos_id, -1)]
                    changdu = len(golds)
                    preds = [total_phonemes[one] for one in predss_l2r[n].cpu().numpy()[:(changdu+1)] if one not in (sos_id, eos_id, -1)]

                    pred_txt.append(''.join(preds))
                    pred_phonemes.append(preds)
                    
                    gold_txt.append(''.join(golds))
                    gold_phonemes.append(golds)

                    pred_all_txt.extend(pred_txt)
                    gold_all_txt.extend(gold_txt)
                    
                for n in range(length_r2l):
                    golds = [total_phonemes[one] for one in gold_r2l[n].cpu().numpy() if one not in (sos_id, eos_id, -1)]
                    changdu = len(golds)
                    preds = [total_phonemes[one] for one in predss_r2l[n].cpu().numpy()[:(changdu+1)] if one not in (sos_id, eos_id, -1)]
                    pred_txt_r2l.append(''.join(preds))
                    pred_phonemes_r2l.append(preds)
                    
                    gold_txt_r2l.append(''.join(golds))
                    gold_phonemes_r2l.append(golds)
                    
                    pred_all_txt_r2l.extend(pred_txt_r2l)
                    gold_all_txt_r2l.extend(gold_txt_r2l)

    l2r_wer = wer_compute(pred_all_txt, gold_all_txt)
    l2r_per = per_compute(pred_phonemes, gold_phonemes)
    
    r2l_wer = wer_compute(pred_all_txt_r2l, gold_all_txt_r2l)
    r2l_per = per_compute(pred_phonemes_r2l, gold_phonemes_r2l)
    print('l2r_wer: ', l2r_wer, 'l2r_per: ', l2r_per)
    print('r2l_wer: ', r2l_wer, 'r2l_per: ', r2l_per)
    
    return l2r_wer, l2r_per, r2l_wer, r2l_per

def valid_lrw1000(valid_loader, model, logger):
    model = model.module
    model.eval()

    pred_all_txt = []
    gold_all_txt = []
    
    pred_phonemes = []
    gold_phonemes = []
    
    pred_all_txt_r2l = []
    gold_all_txt_r2l = []
    
    pred_phonemes_r2l = []
    gold_phonemes_r2l = []
    # Batches
    wer = float(0)
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input, padded_target, padded_target_reverse, _ = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        padded_target_reverse = padded_target_reverse.to(device)

        with torch.no_grad():
                # Forward prop.
                gold_l2r = padded_target
                gold_r2l = padded_target_reverse
                pred_l2r, pred_r2l = model.recognize(padded_input)

                predss_l2r = pred_l2r
                predss_r2l = pred_r2l

                pred_txt = []
                gold_txt = []
                pred_txt_r2l = []
                gold_txt_r2l = []
                length = predss_l2r.size(0)
                length_r2l = predss_r2l.size(0)

                for n in range(length):
                    golds = [total_phonemes[one] for one in gold_l2r[n].cpu().numpy() if one not in (sos_id, eos_id, -1)]
                    changdu = len(golds)
                    preds = [total_phonemes[one] for one in predss_l2r[n].cpu().numpy()[:(changdu+1)] if one not in (sos_id, eos_id, -1)]
                    pred_txt.append(''.join(preds))
                    pred_phonemes.append(preds)
                    
                    gold_txt.append(''.join(golds))
                    gold_phonemes.append(golds)
                    
                    pred_all_txt.extend(pred_txt)
                    gold_all_txt.extend(gold_txt)
                    
                for n in range(length_r2l):
                    golds = [total_phonemes[one] for one in gold_r2l[n].cpu().numpy() if one not in (sos_id, eos_id, -1)]
                    changdu = len(golds)
                    preds = [total_phonemes[one] for one in predss_r2l[n].cpu().numpy()[:(changdu+1)] if one not in (sos_id, eos_id, -1)]
                    pred_txt_r2l.append(''.join(preds))
                    pred_phonemes_r2l.append(preds)
                    
                    gold_txt_r2l.append(''.join(golds))
                    gold_phonemes_r2l.append(golds)
                    
                    pred_all_txt_r2l.extend(pred_txt_r2l)
                    gold_all_txt_r2l.extend(gold_txt_r2l)

    l2r_wer = wer_compute(pred_all_txt, gold_all_txt)
    l2r_per = per_compute(pred_phonemes, gold_phonemes)
    
    r2l_wer = wer_compute(pred_all_txt_r2l, gold_all_txt_r2l)
    r2l_per = per_compute(pred_phonemes_r2l, gold_phonemes_r2l)
    print('l2r_wer: ', l2r_wer, 'l2r_per: ', l2r_per)
    print('r2l_wer: ', r2l_wer, 'r2l_per: ', r2l_per)

    return l2r_wer, l2r_per, r2l_wer, r2l_per

def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
