import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
# from torch import nn
from tqdm import tqdm
import editdistance
from cvtransforms import *

from config import device, print_freq, sos_id, eos_id, word_number, p
from data_gen import AiShellDataset, pad_collate, get_lrw_labeled_and_lrw1000_labeled_idxs, TwoStreamBatchSampler
from data_gen_LRW import AiShellDatasetLRW

from transformer.encoder import Encoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

import pickle

#from list_vocabs import words

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
        encoder_v = Encoder(512, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)

        model = Transformer(encoder_v, pt)
        # print(model)
        #model = nn.DataParallel(model, device_ids=[0,1,2])

        # optimizer
        optimizer = TransformerOptimizer(
            torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.98), eps=1e-09))

    else:
        encoder_v = Encoder(512, args.n_layers_enc, args.n_head,
                          args.d_k, args.d_v, args.d_model, args.d_inner,
                          dropout=args.dropout, pe_maxlen=args.pe_maxlen)
      
        model = Transformer(encoder_v, pt)
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
        model.train()  # train mode (dropout and batchnorm is used)
        losses = AverageMeter()
        # Batches
        running_corrects_train_visual_lrw = 0
        running_all_train_visual_lrw = 0
        running_corrects_train_visual_lrw1000 = 0
        running_all_train_visual_lrw1000 = 0

        for i, (data) in enumerate(train_loader):
            # Move to GPU, if available
            padded_input_visual, _, padded_target, indictions = data
            #print(padded_input_visual.size())
            #print(indictions)
            #batch_img = RandomCrop(padded_input_visual.numpy(), (88, 88))
            #batch_img = ColorNormalize(padded_input_visual.numpy())
            #padded_input = torch.from_numpy(batch_img)
            #print(padded_input_visual.size())
            padded_input = torch.FloatTensor(padded_input_visual.float())

            padded_input_visual = padded_input.to(device)
            padded_target = padded_target.to(device)
            languages = indictions.to(device)
            
            v_t, v_t_languages = model(padded_input_visual)
            #print(v_t_lrw.size(), v_t_lrw1000.size())
            _, visual_preds = torch.max(F.softmax(v_t, dim=1).data, 1)
            _, visual_preds_lrw1000 = torch.max(F.softmax(v_t_languages, dim=1).data, 1)

            running_corrects_train_visual_lrw += torch.sum(visual_preds == padded_target.data)
            running_all_train_visual_lrw += len(padded_input_visual)
            
            running_corrects_train_visual_lrw1000 += torch.sum(visual_preds_lrw1000 == languages.data)
            running_all_train_visual_lrw1000 += len(padded_input_visual)
            #print(visual_preds_lrw)
            #print(padded_target[:args.lrw_batch_size])

            loss_1500 = criterion(v_t, padded_target)
            loss_2 = criterion(v_t_languages, languages)
            
            loss = loss_1500 + 0.1* loss_2
            #loss = loss_lrw
            #loss = loss_lrw1000
            # Back prop.
            
            optimizer.zero_grad()
            loss.backward()

            # Update weights
            optimizer.step()
            
            losses.update(loss.item())

            # Print status
            if i % print_freq == 0:
                logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Loss (loss:{3}, loss_1500:{4}, loss_2:{5}) {loss.val:.5f} ({loss.avg:.5f}) '.format(epoch, i, len(train_loader), loss.item(), loss_1500.item(), loss_2.item(), loss=losses))

            #print('train_loss: ', train_loss)
            writer.add_scalar('model_{}/v_t_train_loss'.format(p), loss_1500, k)
            writer.add_scalar('model_{}/v_t_languages_train_loss'.format(p), loss_2, k)
          
            k += 1
            #if k == 1:
             #   break
        
        visual_train_acc_lrw = running_corrects_train_visual_lrw.item() / running_all_train_visual_lrw
        visual_train_acc_lrw1000 = running_corrects_train_visual_lrw1000.item() / running_all_train_visual_lrw1000
        #audio_train_acc = running_corrects_train_audio.item() / running_all_train_audio
        print('the word classifier accuracy: ',visual_train_acc_lrw, 'the language classifier accuracy: ', visual_train_acc_lrw1000)
        
        writer.add_scalar('model_{}/words_train_acc'.format(p), visual_train_acc_lrw, epoch)
        writer.add_scalar('model_{}/languages_train_acc'.format(p), visual_train_acc_lrw1000, epoch)
        lr = optimizer.lr
        
        print('\nLearning rate: {}'.format(lr))
        writer.add_scalar('model_{}/learning_rate'.format(p), lr, epoch)
        step_num = optimizer.step_num
        
        print('Step num: {}\n'.format(step_num))

        # One epoch's validation
        v_acc_lrw = 0
        visual_valid_loss_lrw, v_acc_lrw, acc1 = valid_lrw(valid_loader=valid_loader_LRW, model=model,criterion = criterion, logger=logger, batch_size=args.batch_size)
                      
        visual_valid_loss_lrw1000, v_acc_lrw1000, acc2 = valid_lrw1000(valid_loader=valid_loader_LRW1000, model=model, criterion = criterion, logger=logger, batch_size=args.batch_size-args.batch_size)

        writer.add_scalar('model_{}/visual_valid_acc_lrw'.format(p), v_acc_lrw, epoch)
        writer.add_scalar('model_{}/visual_valid_acc_lrw1000'.format(p), v_acc_lrw1000, epoch)
        writer.add_scalar('model_{}/visual_valid_languages_acc_lrw'.format(p), acc1, epoch)
        writer.add_scalar('model_{}/visual_valid_languages_acc_lrw1000'.format(p), acc2, epoch)
        writer.add_scalar('model_{}/visual_valid_loss_lrw'.format(p), visual_valid_loss_lrw, epoch)
        writer.add_scalar('model_{}/visual_valid_loss_lrw1000'.format(p), visual_valid_loss_lrw1000, epoch)
        # Check if there was an improvement
        acc = v_acc_lrw+v_acc_lrw1000
        is_best = (1-acc) < best_loss
        best_loss = min((1-acc), best_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, model, optimizer, best_loss, is_best)


def valid_lrw(valid_loader, model, criterion, logger, batch_size):
    model.eval()

    losses1 = AverageMeter()
    losses2 = AverageMeter()

    running_corrects_val_visual = 0
    running_all_val_visual = 0
    running_corrects_train_visual_lrw1000 = 0
    running_all_train_visual_lrw1000 = 0
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input_visual, padded_input_audio, padded_target, indictions = data
        
        batch_img = CenterCrop(padded_input_visual.numpy(), (88, 88))
        batch_img = ColorNormalize(batch_img)
        
        padded_input = torch.from_numpy(batch_img)
        #padded_inputs = inputs.float().permute(0, 4, 1, 2, 3).contiguous()
        padded_input = torch.FloatTensor(padded_input.float())
        padded_input_visual = padded_input.to(device)
        padded_target = padded_target.to(device)
        languages = indictions.to(device)

        with torch.no_grad():
            # Forward prop.
            v_t_lrw, v_t_languages = model(padded_input_visual)
            outputs1 = v_t_lrw
            
            _, visual_preds = torch.max(F.softmax(v_t_lrw, dim=1).data, 1)
            _, visual_preds_lrw1000 = torch.max(F.softmax(v_t_languages, dim=1).data, 1)

            running_corrects_val_visual += torch.sum(visual_preds == padded_target.data)
            running_all_val_visual += len(padded_input_visual)
            
            running_corrects_train_visual_lrw1000 += torch.sum(visual_preds_lrw1000 == languages.data)
            running_all_train_visual_lrw1000 += len(padded_input_visual)

            loss1 = criterion(outputs1, padded_target)

        losses1.update(loss1.item())
   
        # Print status
    #logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    v_acc = running_corrects_val_visual.item() / running_all_val_visual
    languages_acc = running_corrects_train_visual_lrw1000.item() / running_all_train_visual_lrw1000
    
    print('lrw val accuracy: ', v_acc, 'lrw_language_acc: ', languages_acc)

    return losses1.avg, v_acc, languages_acc

def valid_lrw1000(valid_loader, model, criterion, logger, batch_size):
    model.eval()

    losses1 = AverageMeter()
    
    running_corrects_val_visual = 0
    running_all_val_visual = 0
    running_corrects_train_visual_lrw1000 = 0
    running_all_train_visual_lrw1000 = 0
    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input_visual, padded_input_audio, padded_target, indictions = data
        padded_input_visual = padded_input_visual.to(device)
        #padded_input_audio = padded_input_audio.to(device)
        padded_target = padded_target.to(device)
        languages = indictions.to(device)

        with torch.no_grad():
            # Forward prop.
            v_t_lrw, v_t_languages = model(padded_input_visual)
            outputs1 = v_t_lrw
            
            _, visual_preds = torch.max(F.softmax(v_t_lrw, dim=1).data, 1)
            _, visual_preds_lrw1000 = torch.max(F.softmax(v_t_languages, dim=1).data, 1)

            running_corrects_val_visual += torch.sum(visual_preds == padded_target.data)
            running_all_val_visual += len(padded_input_visual)
            
            running_corrects_train_visual_lrw1000 += torch.sum(visual_preds_lrw1000 == languages.data)
            running_all_train_visual_lrw1000 += len(padded_input_visual)

            loss1 = criterion(outputs1, padded_target)

        losses1.update(loss1.item())
   
        # Print status
    #logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    v_acc = running_corrects_val_visual.item() / running_all_val_visual
    languages_acc = running_corrects_train_visual_lrw1000.item() / running_all_train_visual_lrw1000
    
    print('lrw1000 val accuracy: ', v_acc, 'lrw1000_language_acc: ', languages_acc)

    return losses1.avg, v_acc, languages_acc

def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()
