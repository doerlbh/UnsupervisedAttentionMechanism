##########################
# Modified from https://github.com/chenzhaomin123/draw_pytorch

# by Baihan Lin for Unsupervised Attention Mechanism
# https://arxiv.org/abs/1902.10658

# Environment: Python 3.6+, Pytorch 1.0
##########################

import torch

# T = 10
# batch_size = 64
# A = 28
# B = 28
# z_size = 10
# N = 5
# dec_size = 256
# enc_size = 256
# epoch_num = 20
# learning_rate = 1e-3
# beta1 = 0.5
# USE_CUDA = True
# clip = 5.0

T = 5
batch_size = 30
A = 28
B = 28
z_size = 10
N = 5
dec_size = 10
enc_size = 10
epoch_num = 5
learning_rate = 1e-3
beta1 = 0.5
USE_CUDA = True
clip = 5.0

if USE_CUDA:
    device = 'cuda'
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = 'cpu'
    torch.set_default_tensor_type('torch.FloatTensor')