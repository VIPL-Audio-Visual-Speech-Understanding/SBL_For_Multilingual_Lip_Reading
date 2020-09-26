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
from data_gen_LRW import AiShellDatasetLRW

from transformer.encoder import Encoder
from transformer.decoder import Decoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

import pickle

word_length = 14

english_id = 2
chinese_id = 3
#from list_vocabs import words
#total_phonemes = ['sos', 'eos', 's', 'p', 'ii', 'k', 'i', 'ng', 'l', 'e', 'v', 'e1', 'a1', 'm', 'z', 'zh', 'o', 'r', 'eu', 't', 'ai', 'h', 'th', 'y', 'n', 'ch', 'ae', 'au', 'er', 'd', 'f', 'ei', 'w', 'a', 'oi', 'b', 'uu', 'g', 'sh', 'dh', 'u', 'zh1', 'an', 'ang', 'en', 'eng', 'ie', 'in', 'ing', 'uo', 'ts', 'iii', 'ong', 'j', 'yu', 'yue', 'q', 'x']

total_phonemes = ['sos', 'eos', 'english', 'chinese', 's', 'p', 'ii', 'k', 'i', 'ng', 'l', 'e', 'v', 'e1', 'a1', 'm', 'z', 'zh', 'o', 'r', 'eu', 't', 'ai', 'h', 'th', 'y', 'n', 'ch', 'ae', 'au', 'er', 'd', 'f', 'ei', 'w', 'a', 'oi', 'b', 'uu', 'g', 'sh', 'dh', 'u', 'zh1', 'an', 'ang', 'en', 'eng', 'ie', 'in', 'ing', 'uo', 'ts', 'iii', 'ong', 'j', 'yu', 'yue', 'q', 'x']

print(total_phonemes)
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
    #pt = 'acc0.84412.pt'
    pt='acc0.38423358796386053.pt'
    #pt=None
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
    model = nn.DataParallel(model, device_ids=[0,1])
    # define a criterion 
    criterion = nn.CrossEntropyLoss()
    criterion_KD = nn.KLDivLoss()

    # Custom dataloaders
    train_dataset = AiShellDataset(args, 'train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler, pin_memory=True, num_workers=args.num_workers)
    
    ###, collate_fn=pad_collate
    valid_dataset_LRW = AiShellDatasetLRW(args, 'val')
    valid_loader_LRW = torch.utils.data.DataLoader(valid_dataset_LRW, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    valid_dataset_LRW1000 = AiShellDataset(args, 'val')
    valid_loader_LRW1000 = torch.utils.data.DataLoader(valid_dataset_LRW1000, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

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
        writer.add_scalar('model_{}/train_loss'.format(word_length), train_loss, epoch)

        lr = optimizer.lr
        print('\nLearning rate: {}'.format(lr))
        writer.add_scalar('model_{}/learning_rate'.format(word_length), lr, epoch)
        step_num = optimizer.step_num
        print('Step num: {}\n'.format(step_num))
        
        state = {'model': model}
        filename = 'BEST_current_checkpoint.tar'
        torch.save(state, filename)
        #save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)
        '''
        # One epoch's validation
        valid_loss_lrw, wer_lrw, per_lrw = valid_lrw(valid_loader=valid_loader_LRW,model=model,logger=logger)
        valid_loss_lrw1000, wer_lrw1000, per_lrw1000 = valid_lrw1000(valid_loader=valid_loader_LRW1000,model=model,logger=logger)
        writer.add_scalar('model_{}/valid_loss_lrw'.format(word_length), valid_loss_lrw, epoch)
        writer.add_scalar('model_{}/valid_wer_lrw'.format(word_length), wer_lrw, epoch)
        writer.add_scalar('model_{}/valid_per_lrw'.format(word_length), per_lrw, epoch)

        writer.add_scalar('model_{}/valid_loss_lrw1000'.format(word_length), valid_loss_lrw1000, epoch)
        writer.add_scalar('model_{}/valid_wer_lrw1000'.format(word_length), wer_lrw1000, epoch)
        writer.add_scalar('model_{}/valid_per_lrw1000'.format(word_length), per_lrw1000, epoch)
        # Check if there was an improvement
        is_best = (wer_lrw + wer_lrw1000) < best_loss
        #is_best = train_loss < best_loss
        best_loss = min((wer_lrw + wer_lrw1000), best_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)
         '''   
            

def train(train_loader, model, optimizer, epoch, logger, k):
    #writer = SummaryWriter()
    
    model.train()  # train mode (dropout and batchnorm is used)

    losses = AverageMeter()
    # Batches
    n = k
    for i, (data) in enumerate(train_loader):
        # Move to GPU, if available
        padded_input, padded_target, _ = data
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        pred, gold = model(padded_input, padded_target)
        preds = pred.argmax(-1)
        #print('preds: ', preds)
        #print('golds: ', gold)
        #print(pred.size(), gold.size())
        #print(pred.size(), gold.size())
        loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)

        # Back prop.
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

        #if n%100 == 0:
         #   break

    return losses.avg, n

def valid_lrw(valid_loader, model, logger):
    model.eval()

    losses = AverageMeter()
    pred_all_txt = []
    gold_all_txt = []
    
    pred_phonemes = []
    gold_phonemes = []
    # Batches
    wer = float(0)
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input, padded_target, _ = data
        batch_img = CenterCrop(padded_input.numpy(), (88, 88))
        batch_img = ColorNormalize(batch_img)
        
        padded_input = torch.from_numpy(batch_img)
        #padded_inputs = inputs.float().permute(0, 4, 1, 2, 3).contiguous()
        padded_input = torch.FloatTensor(padded_input.float())
        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        #input_lengths = input_lengths.to(device)
        #if padded_target.size(1) <= word_length:
        with torch.no_grad():
                # Forward prop.
                pred, gold = model(padded_input, padded_target)
                loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
                pred_argmax = pred.argmax(-1)
                predss = pred_argmax
                
                pred_txt = []
                gold_txt = []
                length = predss.size(0)
                for n in range(length):
                #changdu = len(gold[n].cpu().numpy())
                    #print(gold[n])
                    golds = [total_phonemes[one] for one in gold[n].cpu().numpy() if one not in (sos_id, eos_id, english_id, chinese_id, -1)]
                    changdu = len(golds)
                    #print(preds[n].cpu())
                    preds = [total_phonemes[one] for one in predss[n].cpu().numpy()[:changdu] if one not in (sos_id, eos_id, english_id, chinese_id, -1)]
                    pred_txt.append(''.join(preds))
                    pred_phonemes.append(preds)
                    
                    gold_txt.append(''.join(golds))
                    gold_phonemes.append(golds)
                    
                    #print('pred: ', preds)
                    #print('gold: ', golds)
                    
                    pred_all_txt.extend(pred_txt)
                    gold_all_txt.extend(gold_txt)
                
                #print(' '.join(golds))
                #print(pred_argmax, gold)

        # Keep track of metrics
        losses.update(loss.item())

    wer = wer_compute(pred_all_txt, gold_all_txt)
    per = per_compute(pred_phonemes, gold_phonemes)
    print('wer: ', wer, 'per: ', per)
    
    # Print status
    logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    return losses.avg, wer, per

def valid_lrw1000(valid_loader, model, logger):
    model.eval()

    losses = AverageMeter()
    
    pred_all_txt = []
    gold_all_txt = []
    
    pred_phonemes = []
    gold_phonemes = []
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input_visual, padded_target, indictions = data
        padded_input_visual = padded_input_visual.to(device)
        padded_target = padded_target.to(device)

        with torch.no_grad():
                pred, gold = model(padded_input_visual, padded_target)
                loss, n_correct = cal_performance(pred, gold, smoothing=args.label_smoothing)
                pred_argmax = pred.argmax(-1)
                predss = pred_argmax
                #print(preds, preds.cpu().numpy())
                pred_txt = []
                gold_txt = []
                length = predss.size(0)
                for n in range(length):
                #changdu = len(gold[n].cpu().numpy())
                    golds = [total_phonemes[one] for one in gold[n].cpu().numpy() if one not in (sos_id, eos_id, english_id, chinese_id, -1)]
                    changdu = len(golds)
                    preds = [total_phonemes[one] for one in predss[n].cpu().numpy()[:changdu] if one not in (sos_id, eos_id, english_id, chinese_id, -1)]
                    pred_txt.append(''.join(preds))
                    pred_phonemes.append(preds)
                    
                    gold_txt.append(''.join(golds))
                    gold_phonemes.append(golds)
                    
                    #print('pred: ', preds)
                    #print('gold: ', golds)
                    
                    pred_all_txt.extend(pred_txt)
                    gold_all_txt.extend(gold_txt)

        # Keep track of metrics
        losses.update(loss.item())

    wer = wer_compute(pred_all_txt, gold_all_txt)
    per = per_compute(pred_phonemes, gold_phonemes)
    print('wer: ', wer, 'per: ', per)
    
    # Print status
    logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    return losses.avg, wer, per

def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
