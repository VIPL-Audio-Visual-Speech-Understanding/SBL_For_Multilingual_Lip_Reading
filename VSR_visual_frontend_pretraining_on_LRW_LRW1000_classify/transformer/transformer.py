import torch
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
        self.fc_1500 = nn.Linear(512, 1500)
        self.fc_2 = nn.Linear(512, 2)
        
    def forward(self, padded_input_visual):
        """
        Args:
            padded_input: N x Ti x D
            input_lengths: N
            padded_targets: N x To
        """
        padded_input = padded_input_visual.view(padded_input_visual.size(0), -1,padded_input_visual.size(1), padded_input_visual.size(2), padded_input_visual.size(3))
        padded_input = self.visual_frontend(padded_input)
        length = padded_input.size(1)
        batch = padded_input.size(0)
        input_lengths = [len(padded_input[i]) for i in range(batch)]    
        visual_encoder_padded_outputs1, *_ = self.encoder_v(padded_input, input_lengths)

        #visual_encoder_padded_outputs = visual_encoder_padded_outputs1[:, 29, :]
        visual_encoder_padded_outputs = torch.mean(visual_encoder_padded_outputs1, dim=2, keepdim=True)
        visual_encoder_languages = visual_encoder_padded_outputs1[:, 30, :]

        v_t = self.fc_1500(visual_encoder_padded_outputs)
        v_t_languages = self.fc_2(visual_encoder_languages)     

        return v_t, v_t_languages
        



