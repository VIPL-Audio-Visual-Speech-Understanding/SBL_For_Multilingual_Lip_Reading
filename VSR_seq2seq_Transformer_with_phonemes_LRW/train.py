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
from xer import cer_function, wer_function
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence

import pickle
pklfile = 'char_list.pkl'
with open(pklfile, 'rb') as file:
    data = pickle.load(file)
char_list = data

#torch.cuda.set_device(0)
#os.environ['CUDA_VISIBLE_DEVICES']='2,3,0'

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

        optimizer = TransformerOptimizer(torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    logger = get_logger()

    # Move to GPU, if available
    #model = model.to(device)
    #print(model)
    #model = model.module
    #model = model.cuda()
    model = nn.DataParallel(model, device_ids=[0])
    #print(model)
    model = model.to(device)

    # Custom dataloaders
    train_dataset = AiShellDataset(args, 'train')
    #print(train_dataset[0][0].size())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)
    valid_dataset = AiShellDataset(args, 'val')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    # Epochs
    k = 0
    for epoch in range(start_epoch, 1):
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
        wer, per = valid(valid_loader=valid_loader,model=model,logger=logger)
        #writer.add_scalar('model_{}/valid_loss'.format(word_length), valid_loss, epoch)
        writer.add_scalar('model_{}/valid_wer'.format(word_length), wer, epoch)
        writer.add_scalar('model_{}/valid_per'.format(word_length), per, epoch)

        # Check if there was an improvement
        is_best = wer < best_loss
        #is_best = train_loss < best_loss
        best_loss = min(wer, best_loss)
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
        padded_input, padded_target = data
        #padded_input, padded_target = data

        padded_input = padded_input.to(device)
        padded_target = padded_target.to(device)
        pred, gold = model(padded_input, padded_target)
        #pred = pred.argmax(-1)
        
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

        if n%100 == 0:
            break

    return losses.avg, n


def valid(valid_loader, model, logger):
    model = model.module
    #print(model)
    #if isinstance(model, torch.nn.DataParallel):
     #   model = model.module.module.module
    #model = nn.DataParallel(model, device_ids=[0])
    model.eval()
    #print(model)
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
        
        nbest_hyps = nbest_hyps.cpu().numpy()
        #print(padded_target, len(padded_target))
        #print(nbest_hyps, len(nbest_hyps))
        
        for n in range(len(nbest_hyps)):
                #changdu = len(gold[n].cpu().numpy())
                golds = [char_list[one] for one in padded_target[n] if one not in (sos_id, eos_id, -1)]
                changdu = len(golds)
                preds = [char_list[one] for one in nbest_hyps[n][:(changdu+1)] if one not in (sos_id, eos_id, -1)]
                pred_phonemes.append(preds)
                gold_phonemes.append(golds)
                   
                pred_all_txt.append(''.join(preds))
                gold_all_txt.append(''.join(golds))
                
                #cer = cer_function(golds, preds)
                #total_cer += cer
       
    avg_per = per_compute(pred_phonemes, gold_phonemes)
    avg_wer = wer_function(pred_all_txt, gold_all_txt)
    print('avg_per: ' + str(avg_per))
    print('avg_wer: ' + str(avg_wer))
    
    return avg_per, avg_wer



def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
