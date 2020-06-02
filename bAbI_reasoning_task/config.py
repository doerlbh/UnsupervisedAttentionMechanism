##########################
# Modified from https://github.com/chingyaoc/ggnn.pytorch

# by Baihan Lin for Unsupervised Attention Mechanism
# https://arxiv.org/abs/1902.10658

# Environment: Python 3.6, Pytorch 1.0
##########################

# config.py maintains those hyperparameters and information that don't change across different tasks

NUM_WORKER = 4
BATCH_SIZE = 10
# LEARNING_RATE = 0.001
VERBAL = True # whether print logs or not
MODEL_PATH = '' # the model we want to resume
ANNOTATION_DIM = {str(4):1, str(15):1, str(16):1, str(18):2, str(19):2}

device = "cuda"
