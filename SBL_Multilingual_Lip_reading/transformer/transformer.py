import torch.nn as nn
import torch.nn.functional as F
from .video_frontend import visual_frontend

class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, decoder, pt):
        super(Transformer, self).__init__()
        self.visual_frontend = visual_frontend(pt)
        self.encoder = encoder
        self.decoder = decoder

        # for p in self.parameters():
        #    p.requires_grad = False

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, padded_input, padded_target_l2r, padded_target_r2l):

        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        #print(padded_input_visual.shape)
        padded_input = padded_input.unsqueeze(4)    ###gray
        #print(padded_input.size())
        padded_input = padded_input.permute(0,4,1,2,3)
        padded_input = self.visual_frontend(padded_input)
        length = padded_input.size(1)
        batch = padded_input.size(0)
        input_lengths = [len(padded_input[i]) for i in range(batch)]    
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        
        
        pred_l2r, gold_l2r, pred_r2l, gold_r2l = self.decoder(padded_target_l2r, padded_target_r2l, encoder_padded_outputs, input_lengths)
        
        return pred_l2r, gold_l2r, pred_r2l, gold_r2l

    def recognize(self, input):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        input = input.unsqueeze(4)
        input = input.permute(0,4,1,2,3)
        #print(input.size())
        input = self.visual_frontend(input)
        #print(input.size())
        length = input.size(1)
        batch = input.size(0)
        #print('length is: ', length)
        input_lengths = [len(input[i]) for i in range(batch)]

        #input_length = input.size(1)

        #encoder_outputs, *_ = self.encoder(input.unsqueeze(0), input_lengths)
        encoder_outputs, *_ = self.encoder(input, input_lengths)
        ys_l2r, ys_r2l = self.decoder.recognize_beam(encoder_outputs)
        return ys_l2r, ys_r2l