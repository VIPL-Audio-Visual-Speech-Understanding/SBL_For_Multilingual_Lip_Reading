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
        
    def forward(self, padded_input, padded_target):

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
        
        
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)
        #print(pred.size(), gold.size())
        return pred, gold

    def recognize(self, input, char_list, args):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        input = input.unsqueeze(0)
        input = input.permute(0,4,1,2,3)
        #print(input.size())
        input = self.lipreading(input)
        #print(input.size())
        length = input.size(1)
        batch = input.size(0)
        #print('length is: ', length)
        input_lengths = [len(input[i]) for i in range(batch)]

        #input_length = input.size(1)

        #encoder_outputs, *_ = self.encoder(input.unsqueeze(0), input_lengths)
        encoder_outputs, *_ = self.encoder(input, input_lengths)
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs[0],
                                                 char_list,
                                                 args)
        return nbest_hyps


