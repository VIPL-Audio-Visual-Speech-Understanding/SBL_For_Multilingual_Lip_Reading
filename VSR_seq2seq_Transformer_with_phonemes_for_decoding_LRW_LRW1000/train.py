import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
# from torch import nn
from tqdm import tqdm
import editdistance
from cvtransforms import *

from config import device, print_freq, sos_id, eos_id, word_number, p, vocab_size
from data_gen import AiShellDataset, pad_collate
from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

import pickle

word_length = 14
#from list_vocabs import words
lrw1000_phonemes = ['sos', 'eos', 's', 'au', 'm', 'i', 'p', 'ii', 't', 'q', 'yu', 'x', 'j', 'an', 'y', 'eu', 'sh', 'iii', 'd', 'ong', 'ang', 'zh', 'l', 'e1', 'f', 'g', 'eng', 'ts', 'uo', 'a', 'ch', 'w', 'en', 'h', 'u', 'ai', 'yue', 'uu', 'in', 'ing', 'ei', 'z', 'b', 'zh1', 'k', 'ie', 'er', 'n']

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

    writer = SummaryWriter()
    
    epochs_since_improvement = 0
    pt = 'acc0.38423358796386053.pt'
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
        name = checkpoint
        checkpoint = torch.load(checkpoint)
        print('loading {} parameters successful!'.format(name))
        #epochs_since_improvement = checkpoint['epochs_since_improvement']
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
    model = nn.DataParallel(model, device_ids=[0,1])
    # define a criterio
    criterion = nn.CrossEntropyLoss()
    criterion_KD = nn.KLDivLoss()

    # Custom dataloaders
    train_dataset = AiShellDataset(args, 'train')
    #print(train_dataset[0][0].size())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
    ###, collate_fn=pad_collate
    valid_dataset = AiShellDataset(args, 'val')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    # Epochs
    k = 0
    for epoch in range(start_epoch, args.epochs):
        # One epoch's training
        train_loss, n = train(train_loader=train_loader,
                           model=model,
                           optimizer=optimizer,
                           epoch=epoch,
                           logger=logger, k=k)
        k = n
        print('train_loss: ', train_loss)
        writer.add_scalar('model_{}/train_loss'.format(word_length), train_loss, epoch)

        lr = optimizer.lr
        print('\nLearning rate: {}'.format(lr))
        writer.add_scalar('model_{}/learning_rate'.format(word_length), lr, epoch)
        step_num = optimizer.step_num
        print('Step num: {}\n'.format(step_num))

        # One epoch's validation
        wer_l2r, per_l2r, wer_r2l, per_r2l = valid(valid_loader=valid_loader,model=model,logger=logger)
        #writer.add_scalar('model_{}/valid_loss_l2r'.format(word_length), valid_loss_l2r, epoch)
        writer.add_scalar('model_{}/valid_wer_l2r'.format(word_length), wer_l2r, epoch)
        writer.add_scalar('model_{}/valid_per_l2r'.format(word_length), per_l2r, epoch)
        
        #writer.add_scalar('model_{}/valid_loss_r2l'.format(word_length), valid_loss_r2l, epoch)
        writer.add_scalar('model_{}/valid_wer_r2l'.format(word_length), wer_r2l, epoch)
        writer.add_scalar('model_{}/valid_per_r2l'.format(word_length), per_r2l, epoch)

        # Check if there was an improvement
        is_best = (wer_l2r+wer_r2l) < best_loss
        #is_best = train_loss < best_loss
        best_loss = min((wer_l2r+wer_r2l), best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def train(train_loader, model, optimizer, epoch, logger, k):
    #writer = SummaryWriter()
    
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    # Batches
    n = k
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        padded_input, padded_target, padded_target_reverse = data
        #padded_input, padded_target = data
        #print(padded_input.size(), padded_target.size())
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        padded_target_reverse = padded_target_reverse.to(device)
        
        pred_l2r, gold_l2r, pred_r2l, gold_r2l = model(padded_input, padded_target, padded_target_reverse)
        #pred = pred.argmax(-1)
        
        #print(pred.size(), gold.size())
        #print(pred.size(), gold.size())
        loss_l2r, n_correct_l2r = cal_performance(pred_l2r, gold_l2r, smoothing=args.label_smoothing)
        loss_r2l, n_correct_r2l = cal_performance(pred_r2l, gold_r2l, smoothing=args.label_smoothing)
        # Back prop.
        
        loss = 0.5*(loss_l2r+loss_r2l)
        #loss = loss_r2l
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()
        
        #writer.add_scalar('model_{}/train_iteration_loss'.format(word_length), loss.item(), n)
        n += 1
        # Keep track of metrics
        losses.update(loss.item())

        # Print status
        if i % print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(train_loader), loss=losses))

        #if n%10== 0:
         #   break

    return losses.avg, n


def valid(valid_loader, model, logger):
    #model = nn.DataParallel(model, device_ids=[0,1,2])
    #model = model.to(device)
    model = model.module
    model.eval()
    #print(model)
    #losses_l2r = AverageMeter()
    #losses_r2l = AverageMeter()
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
        padded_input, padded_target, padded_target_reverse = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        padded_target_reverse = padded_target_reverse.to(device)
        #input_lengths = input_lengths.to(device)
        #if padded_target.size(1) <= word_length:
        with torch.no_grad():
                # Forward prop.
                gold_l2r = padded_target
                gold_r2l = padded_target_reverse
                pred_l2r, pred_r2l = model.recognize(padded_input)
                #print(pred_l2r)
                #print(gold_l2r)
                #loss_l2r, n_correct_l2r = cal_performance(pred_l2r, gold_l2r, smoothing=args.label_smoothing)
                #loss_r2l, n_correct_r2l = cal_performance(pred_r2l, gold_r2l, smoothing=args.label_smoothing)
                
                #pred_l2r_argmax = pred_l2r.argmax(-1)
                predss_l2r = pred_l2r
                
                #pred_r2l_argmax = pred_r2l.argmax(-1)
                predss_r2l = pred_r2l
                
                #print(preds, preds.cpu().numpy())
                pred_txt = []
                gold_txt = []
                pred_txt_r2l = []
                gold_txt_r2l = []
                length = predss_l2r.size(0)
                length_r2l = predss_r2l.size(0)
                for n in range(length):
                #changdu = len(gold[n].cpu().numpy())
                    golds = [lrw1000_phonemes[one] for one in gold_l2r[n].cpu().numpy() if one not in (sos_id, eos_id, -1)]
                    changdu = len(golds)
                    preds = [lrw1000_phonemes[one] for one in predss_l2r[n].cpu().numpy()[:(changdu+1)] if one not in (sos_id, eos_id, -1)]
                    pred_txt.append(''.join(preds))
                    pred_phonemes.append(preds)
                    
                    gold_txt.append(''.join(golds))
                    gold_phonemes.append(golds)
                    
                    #print('pred: ', preds)
                    #print('gold: ', golds)
                    
                    pred_all_txt.extend(pred_txt)
                    gold_all_txt.extend(gold_txt)
                    
                for n in range(length_r2l):
                #changdu = len(gold[n].cpu().numpy())
                    golds = [lrw1000_phonemes[one] for one in gold_r2l[n].cpu().numpy() if one not in (sos_id, eos_id, -1)]
                    changdu = len(golds)
                    preds = [lrw1000_phonemes[one] for one in predss_r2l[n].cpu().numpy()[:(changdu+1)] if one not in (sos_id, eos_id, -1)]
                    pred_txt_r2l.append(''.join(preds))
                    pred_phonemes_r2l.append(preds)
                    
                    gold_txt_r2l.append(''.join(golds))
                    gold_phonemes_r2l.append(golds)
                    
                    #print('pred: ', preds)
                    #print('gold: ', golds)
                    
                    pred_all_txt_r2l.extend(pred_txt_r2l)
                    gold_all_txt_r2l.extend(gold_txt_r2l)
                    
        # Keep track of metrics
        #losses_l2r.update(loss_l2r.item())
        #losses_r2l.update(loss_r2l.item())

    l2r_wer = wer_compute(pred_all_txt, gold_all_txt)
    l2r_per = per_compute(pred_phonemes, gold_phonemes)
    
    r2l_wer = wer_compute(pred_all_txt_r2l, gold_all_txt_r2l)
    r2l_per = per_compute(pred_phonemes_r2l, gold_phonemes_r2l)
    print('l2r_wer: ', l2r_wer, 'l2r_per: ', l2r_per)
    print('r2l_wer: ', r2l_wer, 'r2l_per: ', r2l_per)
    
    # Print status
    #logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    return l2r_wer, l2r_per, r2l_wer, r2l_per

def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()