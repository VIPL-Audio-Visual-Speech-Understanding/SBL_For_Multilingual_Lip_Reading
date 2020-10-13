import os

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # sets device for model and PyTorch tensors

# Model parameters
input_dim = 80  # dimension of feature
window_size = 25  # window size for FFT (ms)
stride = 10  # window stride for FFT (ms)
hidden_size = 512
embedding_dim = 512
cmvn = True  # apply CMVN on feature
num_layers = 4
LFR_m = 4
LFR_n = 3
sample_rate = 16000  # aishell

# Training parameters
grad_clip = 5.  # clip gradients at an absolute value of
print_freq = 50  # print training/validation stats  every __ batches
checkpoint = None  # path to checkpoint, None if none

# Data parameters
IGNORE_ID = -1
sos_id = 0
eos_id = 1
num_train = 120098
num_dev = 14326
num_test = 7176

word_number = 6

####这里的vocab_size要根据具体的实际情况来修�?vocab_size = 500
p = 1
mask = 0.7

pickle_file = 'LRW.pickle'

lrw_path = '../roi_80_116_175_211_npy_gray'
lrw_info = '../LRW_TXT'
lrw_wav = '../lrw_mp4'

lrw1000_path = '../LRW1000_npy_rsz122_gray'
lrw1000_info = '../LRW1000_info'
lrw1000_wav = '../LRW1000_audio'
