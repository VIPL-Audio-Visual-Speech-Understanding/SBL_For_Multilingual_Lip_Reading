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
from data_gen import AiShellDataset, pad_collate
from transformer.encoder import Encoder
from transformer.loss import cal_performance
from transformer.optimizer import TransformerOptimizer
from transformer.transformer import Transformer
from utils import parse_args, save_checkpoint, AverageMeter, get_logger

import pickle

from list_vocabs import words

def train_net(args):
    torch.manual_seed(7)
    np.random.seed(7)
    checkpoint = args.checkpoint
    start_epoch = 0
    best_loss = float('inf')
    epochs_since_improvement = 0
    pt = 'acc0.38423358796386053.pt'
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
        
        checkpoint = torch.load(checkpoint)
        print('loading model parameters successful!')
        #epochs_since_improvement = checkpoint['epochs_since_improvement']
        pretrain_model = checkpoint['model']
        pretrained_dict = pretrain_model.module.state_dict()
        
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}  ###and ('encoder_v' not in k)}  ###
        missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
        print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
        #print('miss matched params:{}'.format(missed_params))              
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
    criterion = nn.CrossEntropyLoss()
    
    valid_dataset = AiShellDataset(args, 'val')
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    test_dataset = AiShellDataset(args, 'test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.num_workers)

    # Epochs
    k = 0
    start_epoch = 0
    for epoch in range(start_epoch, 1):
        # One epoch's training
        losses = AverageMeter()

        # One epoch's validation
        visual_valid_loss, v_acc = valid(valid_loader=valid_loader,
                           model=model,
                           criterion = criterion,
                           logger=logger)

def valid(valid_loader, model, criterion, logger):
    model.eval()

    losses1 = AverageMeter()
    
    running_corrects_val_visual = 0
    running_all_val_visual = 0

    for data in tqdm(valid_loader):
        # Move to GPU, if available
        padded_input_visual, padded_input_audio, padded_target = data
        padded_input_visual = padded_input_visual.to(device)
        #padded_input_audio = padded_input_audio.to(device)
        padded_target = padded_target.to(device)

        with torch.no_grad():
            # Forward prop.
            v_a, v_t = model(padded_input_visual)
            
            outputs1 = v_t
            
            _, visual_preds = torch.max(F.softmax(outputs1, dim=1).data, 1)
        
            running_corrects_val_visual += torch.sum(visual_preds == padded_target.data)
            running_all_val_visual += len(padded_input_visual)

            loss1 = criterion(outputs1, padded_target)

        losses1.update(loss1.item())
       
        # Print status
    #logger.info('\nValidation Loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(loss=losses))

    v_acc = running_corrects_val_visual.item() / running_all_val_visual
    #a_acc = running_corrects_val_audio.item() / running_all_val_audio
    print('val accuracy: ', v_acc)

    return losses1.avg, v_acc


def main():
    global args
    args = parse_args()
    train_net(args)


if __name__ == '__main__':
    main()