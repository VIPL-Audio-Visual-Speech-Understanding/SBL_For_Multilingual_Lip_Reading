import torch.nn as nn
from .video_frontend import Lipreading

class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lipreading = Lipreading(hiddenDim=512, embedSize=256)

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
        #print(padded_input.size())
        padded_input = padded_input.unsqueeze(4)    ###gray
        #print(padded_input.size())
        padded_input = padded_input.permute(0,4,1,2,3)
        #print(padded_input.size())
        padded_input = self.lipreading(padded_input)
        length = padded_input.size(1)
        batch = padded_input.size(0)
        #print('length is: ', length)
        input_lengths = [len(padded_input[i]) for i in range(batch)]
        
        #print(padded_input.size(), input_lengths)
        encoder_padded_outputs, *_ = self.encoder(padded_input, input_lengths)
        # pred is score before softmax
        #print(encoder_padded_outputs.size())
        pred, gold, *_ = self.decoder(padded_target, encoder_padded_outputs,
                                      input_lengths)
        #print(pred.size(), max(pred.size(1)))
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
        #input = input.unsqueeze(0)
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
        #total_preds = []
        encoder_outputs, *_ = self.encoder(input, input_lengths)
        #for n in range(batch):
            #pred = []
        nbest_hyps = self.decoder.recognize_beam(encoder_outputs,
                                                 char_list,
                                                 args)
        #total_preds.append(nbest_hyps[0]['yseq'])
        return nbest_hyps
