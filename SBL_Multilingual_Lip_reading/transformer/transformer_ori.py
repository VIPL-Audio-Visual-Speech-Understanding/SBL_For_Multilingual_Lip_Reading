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
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, padded_input_visual, padded_target):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        #padded_input = padded_input.permute(0,4,1,2,3)
        padded_input = padded_input_visual.view(padded_input_visual.size(0), -1, padded_input_visual.size(1), padded_input_visual.size(2), padded_input_visual.size(3))
        #print(padded_input.shape)
        padded_input = self.visual_frontend(padded_input)
        length = padded_input.size(1)
        batch = padded_input.size(0)
        input_lengths = [len(padded_input[i]) for i in range(batch)]    
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)
        
        return pred, gold
        
    def recognize(self, padded_input_visual):
        """Sequence-to-Sequence beam search, decode one utterence now.
        Args:
            input: T x D
            char_list: list of characters
            args: args.beam
        Returns:
            nbest_hyps:
        """
        #input = input.unsqueeze(0)
        input = padded_input_visual.view(padded_input_visual.size(0), -1, padded_input_visual.size(1), padded_input_visual.size(2), padded_input_visual.size(3))
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
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs)
        return nbest_hyps





