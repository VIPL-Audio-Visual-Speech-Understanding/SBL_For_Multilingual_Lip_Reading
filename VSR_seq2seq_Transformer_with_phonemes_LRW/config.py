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

word_length = 7

p = 1
mask = 0.7
vocab_size = 42

pickle_file = 'LRW_phonemes.pickle'

lrw_path = '/train/roi_80_116_175_211_npy_gray'
lrw_info = '/train/LRW_TXT'
lrw_wav = '/train/lrw_wav/lrw_mp4'

