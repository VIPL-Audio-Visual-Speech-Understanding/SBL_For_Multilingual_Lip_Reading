import torch.nn as nn
import torch.nn.functional as F
from .video_frontend import visual_frontend

class Transformer(nn.Module):
    """An encoder-decoder framework only includes attention.
    """

    def __init__(self, encoder_v, pt):
        super(Transformer, self).__init__()
        self.visual_frontend = visual_frontend(pt)
        self.encoder_v = encoder_v
       
        #for p in self.parameters():
         #   p.requires_grad = False
        #self.fc_lrw_front = nn.Linear(512,512)
        #self.fc_lrw1000_front = nn.Linear(512,512)
        self.fc_1500 = nn.Linear(512, 1500)
        self.fc_2 = nn.Linear(512, 2)
        
    def forward(self, padded_input_visual):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        #padded_input = padded_input.permute(0,4,1,2,3)
        padded_input = padded_input_visual.view(padded_input_visual.size(0), -1,padded_input_visual.size(1), padded_input_visual.size(2), padded_input_visual.size(3))
        #print(padded_input.shape)
        padded_input = self.visual_frontend(padded_input)
        length = padded_input.size(1)
        batch = padded_input.size(0)
        input_lengths = [len(padded_input[i]) for i in range(batch)]    
        visual_encoder_padded_outputs1, *_ = self.encoder_v(padded_input, input_lengths)
        #print(visual_encoder_padded_outputs1.shape)
        visual_encoder_padded_outputs = visual_encoder_padded_outputs1[:, 29, :]
        #print(visual_encoder_padded_outputs.shape)
        visual_encoder_languages = visual_encoder_padded_outputs1[:, 30, :]
        
        #print(visual_encoder_padded_outputs.shape)
        v_t = self.fc_1500(visual_encoder_padded_outputs)
        #print(v_t_lrw.shape, lrw_batch_size, batch_size)
        v_t_languages = self.fc_2(visual_encoder_languages)     
        #print('v_t_lrw_size:', v_t_lrw.shape, 'v_t_lrw1000_size:', v_t_lrw1000.shape)
        return v_t, v_t_languages
        



