import pickle

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import IGNORE_ID, device
from .attention import MultiHeadAttention
from .module import PositionalEncoding, PositionwiseFeedForward
from .utils import get_attn_key_pad_mask, get_attn_pad_mask, get_non_pad_mask, get_subsequent_mask, pad_list

###len(bigram_freq=4335)
#print(len(bigram_freq))

class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self, sos_id, eos_id,
            n_tgt_vocab, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1,
            tgt_emb_prj_weight_sharing=True,
            pe_maxlen=5000):
        super(Decoder, self).__init__()
        # parameters
        self.sos_id = sos_id  # Start of Sentence
        self.eos_id = eos_id  # End of Sentence
        self.n_tgt_vocab = n_tgt_vocab
        
        self.d_word_vec = d_word_vec
        self.n_layers = n_layers
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.tgt_emb_prj_weight_sharing = tgt_emb_prj_weight_sharing
        self.pe_maxlen = pe_maxlen

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(dropout)
        
        self.layer_first_l2r = DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.layer_stack_l2r = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(self.n_layers-1)])
            
        self.layer_first_r2l = DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
        self.layer_stack_r2l = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(self.n_layers-1)])

        self.x_logit_scale = 1.
        ### 58 = 56 + <sos> + <eos>
        self.tgt_word_prj_l2r = nn.Linear(512, 58, bias=False)
        self.tgt_word_prj_r2l = nn.Linear(512, 58, bias=False)
        
    def preprocess(self, padded_input):
        """Generate decoder input and output label from padded_input
        Add <sos> to decoder input, and add <eos> to decoder output label
        """
        ys = [y[y != IGNORE_ID] for y in padded_input]  # parse padded ys
        # prepare input and output word sequences with sos/eos IDs
        eos = ys[0].new([self.eos_id])
        sos = ys[0].new([self.sos_id])
        ys_in = [torch.cat([sos, y], dim=0) for y in ys]
        ys_out = [torch.cat([y, eos], dim=0) for y in ys]
        # padding for ys with -1
        # pys: utt x olen
        ys_in_pad = pad_list(ys_in, self.eos_id)
        ys_out_pad = pad_list(ys_out, self.eos_id)
        assert ys_in_pad.size() == ys_out_pad.size()
        return ys_in_pad, ys_out_pad
    
    def forward(self, padded_input_l2r, padded_input_r2l, encoder_outputs,
                encoder_input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x To
            encoder_padded_outputs: N x Ti x H
        Returns:
        """

        # Get Deocder Input and Output
        ###L2R && R2L
        ys_in_pad_l2r, ys_out_pad_l2r = self.preprocess(padded_input_l2r)
        ys_in_pad_r2l, ys_out_pad_r2l = self.preprocess(padded_input_r2l)
        
        #print(ys_out_pad_l2r)
        
        maxlen = 16
        #print(maxlen)
        # prepare sos
        ys_l2r = torch.ones(encoder_outputs.size(0), 1).fill_(self.sos_id).type_as(encoder_outputs).long().to(device)
        ys_r2l = torch.ones(encoder_outputs.size(0), 1).fill_(self.sos_id).type_as(encoder_outputs).long().to(device)
        
        ys_l2r_outputs = torch.zeros(encoder_outputs.size(0), maxlen, self.n_tgt_vocab).to(device)
        ys_r2l_outputs = torch.zeros(encoder_outputs.size(0), maxlen, self.n_tgt_vocab).to(device)
        #print(ys)
        #print(ys_l2r)
        #print(ys_r2l)
        for i in range(maxlen):
                #last_id = ys.cpu().numpy()[0][-1]
                # -- Prepare masks
                non_pad_mask_l2r = torch.ones_like(ys_l2r).float().unsqueeze(-1)  # 1xix1
                slf_attn_mask_l2r = get_subsequent_mask(ys_l2r)
                
                non_pad_mask_r2l = torch.ones_like(ys_r2l).float().unsqueeze(-1)  # 1xix1
                slf_attn_mask_r2l = get_subsequent_mask(ys_r2l)
                
                
                dec_output_l2r = self.dropout(self.tgt_word_emb(ys_l2r) * self.x_logit_scale +
                                  self.positional_encoding(ys_l2r))
        
                dec_output_r2l = self.dropout(self.tgt_word_emb(ys_r2l) * self.x_logit_scale +
                                  self.positional_encoding(ys_r2l))
                #print(dec_output.size())
                
                dec_output_l2r, _, _ = self.layer_first_l2r(dec_output_l2r, encoder_outputs, non_pad_mask=non_pad_mask_l2r, slf_attn_mask=slf_attn_mask_l2r, dec_enc_attn_mask=None)
        
                dec_output_r2l, _, _ = self.layer_first_r2l(dec_output_r2l, encoder_outputs, non_pad_mask=non_pad_mask_r2l, slf_attn_mask=slf_attn_mask_r2l, dec_enc_attn_mask=None)
                
                l2r_index = [n for n in range(dec_output_l2r.size(1))]
                r2l_index = [n for n in range(dec_output_l2r.size(1)-1,-1,-1)]
                #print(r2l_index)
                #print(l2r_index)
        
                dec_output_left = dec_output_l2r
                dec_output_right = dec_output_r2l
        
        
                for n in range(dec_output_l2r.size(1)):
                    dec_output_left[:,n] = dec_output_l2r[:, l2r_index[n]] + dec_output_r2l[:, r2l_index[n]]

                for n in range(dec_output_l2r.size(1)):
                    dec_output_right[:,n] = dec_output_r2l[:, l2r_index[n]] + dec_output_l2r[:, r2l_index[n]]
                
                dec_output_l2r = dec_output_left
                dec_output_r2l = dec_output_right
        
                for n in range(self.n_layers-1):
                    dec_output_l2r, dec_slf_attn, dec_enc_attn = self.layer_stack_l2r[n](
                              dec_output_l2r, encoder_outputs,
                              non_pad_mask=non_pad_mask_l2r,
                              #slf_attn_mask=slf_attn_mask_l2r,
                              slf_attn_mask=None,
                              dec_enc_attn_mask=None)
                
                    dec_output_r2l, dec_slf_attn, dec_enc_attn = self.layer_stack_r2l[n](
                              dec_output_r2l, encoder_outputs,
                              non_pad_mask=non_pad_mask_r2l,
                              #slf_attn_mask=slf_attn_mask_r2l,
                              slf_attn_mask=None,
                              dec_enc_attn_mask=None)
                              
                    for i in range(dec_output_l2r.size(1)):
                        dec_output_l2r[:,i] = dec_output_l2r[:, l2r_index[i]] + dec_output_r2l[:, r2l_index[i]]

                    for i in range(dec_output_r2l.size(1)):
                        dec_output_r2l[:,i] = dec_output_r2l[:, l2r_index[i]] + dec_output_l2r[:, r2l_index[i]]
                              
                pred_l2r = self.tgt_word_prj_l2r(dec_output_l2r[:,-1])
                pred_r2l = self.tgt_word_prj_r2l(dec_output_r2l[:,-1])
                
                #print(pred_l2r.size())
                ys_l2r_outputs[:, i] = pred_l2r
                ys_r2l_outputs[:, i] = pred_r2l
                
                pred_argmax_l2r = pred_l2r.argmax(-1)
                pred_argmax_r2l = pred_r2l.argmax(-1)
                ###teacher_forcing_rate=0-->0.5, ������
                is_teacher = random.random() > 0.5 
                if is_teacher:
                    pred_argmax_l2r = pred_argmax_l2r.unsqueeze(-1)
                    pred_argmax_r2l = pred_argmax_r2l.unsqueeze(-1)
                else:
                    pred_argmax_l2r = ys_out_pad_l2r[:,i].unsqueeze(-1)
                    pred_argmax_r2l = ys_out_pad_r2l[:,i].unsqueeze(-1)
		            #print(pred_argmax.size())
                #print(ys.size())
                ys_l2r = torch.cat((ys_l2r, pred_argmax_l2r), 1)
                ys_r2l = torch.cat((ys_r2l, pred_argmax_r2l), 1)
                #print(ys, ys.size())
                #if i == 2:
                 #   break

        return ys_l2r_outputs, ys_out_pad_l2r, ys_r2l_outputs, ys_out_pad_r2l
    
    def TM_forward_previous(self, padded_input_l2r, padded_input_r2l, encoder_padded_outputs,
                encoder_input_lengths, return_attns=False):
        """
        Args:
            padded_input: N x To
            encoder_padded_outputs: N x Ti x H
        Returns:
        """

        # Get Deocder Input and Output
        ###L2R
        ys_in_pad_l2r, ys_out_pad_l2r = self.preprocess(padded_input_l2r)
        #print(ys_in_pad_l2r)
        # Prepare masks
        non_pad_mask_l2r = get_non_pad_mask(ys_in_pad_l2r, pad_idx=self.eos_id)
        slf_attn_mask_subseq_l2r = get_subsequent_mask(ys_in_pad_l2r)
        slf_attn_mask_keypad_l2r = get_attn_key_pad_mask(seq_k=ys_in_pad_l2r,
                                                     seq_q=ys_in_pad_l2r,
                                                     pad_idx=self.eos_id)

        slf_attn_mask_l2r = (slf_attn_mask_keypad_l2r + slf_attn_mask_subseq_l2r).gt(0)

        output_length_l2r = ys_in_pad_l2r.size(1)
        
        dec_enc_attn_mask_l2r = get_attn_pad_mask(encoder_padded_outputs,
                                              encoder_input_lengths,
                                              output_length_l2r)
                                              
        ###R2L
        ys_in_pad_r2l, ys_out_pad_r2l = self.preprocess(padded_input_r2l)
        #print(ys_in_pad_l2r)
        # Prepare masks
        non_pad_mask_r2l = get_non_pad_mask(ys_in_pad_r2l, pad_idx=self.eos_id)
        slf_attn_mask_subseq_r2l = get_subsequent_mask(ys_in_pad_r2l)
        slf_attn_mask_keypad_r2l = get_attn_key_pad_mask(seq_k=ys_in_pad_r2l,
                                                     seq_q=ys_in_pad_r2l,
                                                     pad_idx=self.eos_id)

        slf_attn_mask_r2l = (slf_attn_mask_keypad_r2l + slf_attn_mask_subseq_r2l).gt(0)

        output_length_r2l = ys_in_pad_r2l.size(1)
        
        dec_enc_attn_mask_r2l = get_attn_pad_mask(encoder_padded_outputs,
                                              encoder_input_lengths,
                                              output_length_r2l)
                                              
        # Forward
        dec_output_l2r = self.dropout(self.tgt_word_emb(ys_in_pad_l2r) * self.x_logit_scale +
                                  self.positional_encoding(ys_in_pad_l2r))
        
        dec_output_r2l = self.dropout(self.tgt_word_emb(ys_in_pad_r2l) * self.x_logit_scale +
                                  self.positional_encoding(ys_in_pad_r2l))
        #print(dec_output_l2r.size())
        #print(dec_output_r2l.size())
        dec_output_l2r, _, _ = self.layer_first_l2r(dec_output_l2r, encoder_padded_outputs, non_pad_mask=non_pad_mask_l2r, slf_attn_mask=slf_attn_mask_l2r, dec_enc_attn_mask=dec_enc_attn_mask_l2r)
        
        dec_output_r2l, _, _ = self.layer_first_r2l(dec_output_r2l, encoder_padded_outputs, non_pad_mask=non_pad_mask_r2l, slf_attn_mask=slf_attn_mask_r2l, dec_enc_attn_mask=dec_enc_attn_mask_r2l)
        
        #print(dec_output_l2r)
        #print(dec_output_r2l)
        
        l2r_index = [n for n in range(dec_output_l2r.size(1))]
        r2l_index = [n for n in range(dec_output_l2r.size(1)-1,-1,-1)]
        #print(r2l_index)
        #print(l2r_index)
        
        dec_output_left = dec_output_l2r
        dec_output_right = dec_output_r2l
        
        
        for n in range(dec_output_l2r.size(1)):
            dec_output_left[:,n] = dec_output_l2r[:, l2r_index[n]] + dec_output_r2l[:, r2l_index[n]]

        for n in range(dec_output_l2r.size(1)):
            dec_output_right[:,n] = dec_output_r2l[:, l2r_index[n]] + dec_output_l2r[:, r2l_index[n]]
        
        dec_output_l2r = dec_output_left
        dec_output_r2l = dec_output_right
        
        for n in range(self.n_layers-1):
            
            dec_output_l2r, dec_slf_attn, dec_enc_attn = self.layer_stack_l2r[n](
                dec_output_l2r, encoder_padded_outputs,
                non_pad_mask=non_pad_mask_l2r,
                slf_attn_mask=None,
                dec_enc_attn_mask=dec_enc_attn_mask_l2r)
                
            dec_output_r2l, dec_slf_attn, dec_enc_attn = self.layer_stack_r2l[n](
                dec_output_r2l, encoder_padded_outputs,
                non_pad_mask=non_pad_mask_r2l,
                slf_attn_mask=None,
                dec_enc_attn_mask=dec_enc_attn_mask_r2l)
            
            #for n in range(dec_output_l2r.size(1)):
             #   dec_output[:,n] = dec_output_l2r[:, l2r_index[n]] + dec_output_r2l[:, r2l_index[n]]
            for i in range(dec_output_l2r.size(1)):
                dec_output_l2r[:,i] = dec_output_l2r[:, l2r_index[i]] + dec_output_r2l[:, r2l_index[i]]

            for i in range(dec_output_r2l.size(1)):
                dec_output_r2l[:,i] = dec_output_r2l[:, l2r_index[i]] + dec_output_l2r[:, r2l_index[i]]
            
        pred_l2r = self.tgt_word_prj_l2r(dec_output_l2r)
        pred_r2l = self.tgt_word_prj_r2l(dec_output_r2l)
        
        pred_l2r, gold_l2r, pred_r2l, gold_r2l = pred_l2r, ys_out_pad_l2r, pred_r2l, ys_out_pad_r2l

        return pred_l2r, gold_l2r, pred_r2l, gold_r2l
    
    def recognize_beam(self, encoder_outputs):
        #print(encoder_outputs.size())
        maxlen = 16
        #print(maxlen)
        # prepare sos
        ys_l2r = torch.ones(encoder_outputs.size(0), 1).fill_(self.sos_id).type_as(encoder_outputs).long()
        ys_r2l = torch.ones(encoder_outputs.size(0), 1).fill_(self.sos_id).type_as(encoder_outputs).long()
        #print(ys)
        
        for i in range(maxlen):
                #last_id = ys.cpu().numpy()[0][-1]
                # -- Prepare masks
                non_pad_mask_l2r = torch.ones_like(ys_l2r).float().unsqueeze(-1)  # 1xix1
                slf_attn_mask_l2r = get_subsequent_mask(ys_l2r)
                
                non_pad_mask_r2l = torch.ones_like(ys_r2l).float().unsqueeze(-1)  # 1xix1
                slf_attn_mask_r2l = get_subsequent_mask(ys_r2l)
                
                
                dec_output_l2r = self.dropout(self.tgt_word_emb(ys_l2r) * self.x_logit_scale +
                                  self.positional_encoding(ys_l2r))
        
                dec_output_r2l = self.dropout(self.tgt_word_emb(ys_r2l) * self.x_logit_scale +
                                  self.positional_encoding(ys_r2l))
                #print(dec_output.size())
                
                dec_output_l2r, _, _ = self.layer_first_l2r(dec_output_l2r, encoder_outputs, non_pad_mask=non_pad_mask_l2r, slf_attn_mask=slf_attn_mask_l2r, dec_enc_attn_mask=None)
        
                dec_output_r2l, _, _ = self.layer_first_r2l(dec_output_r2l, encoder_outputs, non_pad_mask=non_pad_mask_r2l, slf_attn_mask=slf_attn_mask_r2l, dec_enc_attn_mask=None)
                
                l2r_index = [n for n in range(dec_output_l2r.size(1))]
                r2l_index = [n for n in range(dec_output_l2r.size(1)-1,-1,-1)]
                #print(r2l_index)
                #print(l2r_index)
        
                dec_output_left = dec_output_l2r
                dec_output_right = dec_output_r2l
        
        
                for n in range(dec_output_l2r.size(1)):
                    dec_output_left[:,n] = dec_output_l2r[:, l2r_index[n]] + dec_output_r2l[:, r2l_index[n]]

                for n in range(dec_output_l2r.size(1)):
                    dec_output_right[:,n] = dec_output_r2l[:, l2r_index[n]] + dec_output_l2r[:, r2l_index[n]]
                
                dec_output_l2r = dec_output_left
                dec_output_r2l = dec_output_right
        
                for n in range(self.n_layers-1):
                    dec_output_l2r, dec_slf_attn, dec_enc_attn = self.layer_stack_l2r[n](
                              dec_output_l2r, encoder_outputs,
                              non_pad_mask=non_pad_mask_l2r,
                              #slf_attn_mask=slf_attn_mask_l2r,
                              slf_attn_mask=None,
                              dec_enc_attn_mask=None)
                
                    dec_output_r2l, dec_slf_attn, dec_enc_attn = self.layer_stack_r2l[n](
                              dec_output_r2l, encoder_outputs,
                              non_pad_mask=non_pad_mask_r2l,
                              #slf_attn_mask=slf_attn_mask_r2l,
                              slf_attn_mask=None,
                              dec_enc_attn_mask=None)
                    for i in range(dec_output_l2r.size(1)):
                        dec_output_l2r[:,i] = dec_output_l2r[:, l2r_index[i]] + dec_output_r2l[:, r2l_index[i]]

                    for i in range(dec_output_r2l.size(1)):
                        dec_output_r2l[:,i] = dec_output_r2l[:, l2r_index[i]] + dec_output_l2r[:, r2l_index[i]]
                              
                pred_l2r = self.tgt_word_prj_l2r(dec_output_l2r[:,-1])
                pred_r2l = self.tgt_word_prj_r2l(dec_output_r2l[:,-1])
                
                pred_argmax_l2r = pred_l2r.argmax(-1)
                pred_argmax_r2l = pred_r2l.argmax(-1)
                
                pred_argmax_l2r = pred_argmax_l2r.unsqueeze(-1)
                pred_argmax_r2l = pred_argmax_r2l.unsqueeze(-1)
                #print(pred_argmax.size())
                #print(ys.size())
                ys_l2r = torch.cat((ys_l2r, pred_argmax_l2r), 1)
                ys_r2l = torch.cat((ys_r2l, pred_argmax_r2l), 1)
                #print(ys, ys.size())
                #if i == 2:
                 #   break

        return ys_l2r, ys_r2l
        
class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output *= non_pad_mask

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
