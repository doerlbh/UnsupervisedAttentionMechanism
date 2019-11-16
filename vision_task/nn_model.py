import numbers
from torch.nn.parameter import Parameter
import torch.nn.init
import torch.nn.utils.weight_norm as weightNorm

import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from numpy import linalg as LA
from torchvision.utils import save_image

from torch.optim import lr_scheduler
import pyro
import argparse
import time, os, copy
import numpy as np
import random
from IPython import display

import seaborn as sns
import seaborn as sns

import pandas as pd
import sklearn.metrics as sm

from scipy.stats import norm

from utils import *
from nn_model import *

torch.manual_seed(1)  


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

device = "cpu"
dtype = torch.FloatTensor

# Hyper Parameters

EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 128
TIME_STEP = 5        # rnn time step 
INPUT_SIZE = 784         # rnn input size
WINDOW_WIDTH = 5
WINDOW_SQUARE = WINDOW_WIDTH**2
IS_VISUALIZATION = True 


class RNNRN_UAM(nn.Module):

    def __init__(self,num_classes=10, isTest=False):
        super(RNNRN_UAM, self).__init__()

        self.rnn = nn.RNNCell(         
            input_size=INPUT_SIZE,
            hidden_size=100,
            nonlinearity='tanh'
        )
        self.out = nn.Linear(100, num_classes)
        self.isTest = isTest 
        self.timestep = 5
        
    def RegularityEN(self, x, x_history, COMP):

        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)        
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP)))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        l_nml_x = COMP - sum_p_x_thetax
        x = torch.mul(l_nml_x,x)
        
        return x, COMP, l_nml_x

    def cov(self, m, y=None):
        m = torch.t(m)
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def forward(self, x, x_history_stack=None,COMP_stack=None,inputRN=False, isTest=False):
        
        self.isTest = isTest
        
        x = x.view(-1,INPUT_SIZE)
        
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if inputRN:
            if x_history_stack==None:
                x_history_stats = x_stats
                COMP = torch.tensor(0).type(dtype)
            else:
                if self.isTest:
                    x_history_stats = x_history_stack[0] 
                else:
                    x_history_stats = [(x_history_stack[0][0]*x_history_stack[0][2]+x_stats[0]*x_stats[2])/(x_history_stack[0][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[0][2]-1)*torch.pow(x_history_stack[0][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[0][2]+x_stats[2]-2)),
                                   x_history_stack[0][2]+x_stats[2]]
                COMP = COMP_stack[0]
            x_history_input = x_history_stats
            x, COMP_input, l_input = self.RegularityEN(x,x_history_stats,COMP)
        else:
            COMP_input = -1
            x_history_input = -1
            l_input = -1
        COMP_input = COMP_input
        input_x = x
        
        COMP_rnn,l_rnn,x_history_rnn = [],[],[]
        for t in np.arange(self.timestep):
            if t == 0:
                x = self.rnn(input_x)
            else:
                x = self.rnn(input_x,x) 
            x_stats = [torch.mean(x), torch.std(x), torch.numel(x)]
            if x_history_stack==None:
                x_history_stats = x_stats
                COMP = torch.tensor(0).type(dtype)
            else:
                # if self.isTest:
                #     x_history_stats = x_history_stack[1+t] 
                # else:
                #     x_history_stats = [(x_history_stack[1+t][0]*x_history_stack[1+t][2]+x_stats[0]*x_stats[2])/(x_history_stack[1+t][2]+x_stats[2]),torch.sqrt(((x_history_stack[1+t][2]-1)*torch.pow(x_history_stack[1+t][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[1+t][2]+x_stats[2]-2)),
                #                    x_history_stack[1+t][2]+x_stats[2]]
                # COMP = COMP_stack[1+t]
                if self.isTest:
                    x_history_stats = x_history_stack[1] 
                else:
                    x_history_stats = [(x_history_stack[1][0]*x_history_stack[1][2]+x_stats[0]*x_stats[2])/(x_history_stack[1][2]+x_stats[2]),torch.sqrt(((x_history_stack[1][2]-1)*torch.pow(x_history_stack[1][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[1][2]+x_stats[2]-2)),
                                   x_history_stack[1][2]+x_stats[2]]
                COMP = COMP_stack[1]
                COMP_stack[1] = COMP
                x_history_stack[1] = x_history_stats
            
            x_history_rnn_t = x_history_stats
            COMP_rnn_t = -1
            x, COMP_rnn_t, l_rnn_t = self.RegularityEN(x,x_history_stats,COMP)
            COMP_rnn.append(COMP_rnn_t)
            l_rnn.append(l_rnn_t)
            x_history_rnn.append(x_history_rnn_t)

        x = self.out(x)
        out = F.log_softmax(x, dim=1)
        
        x_history_stack = [x_history_input]
        COMP_stack = [COMP_input]
        MDL = [l_input]
        for i in np.arange(self.timestep):
            x_history_stack.append(x_history_rnn[i])
            COMP_stack.append(COMP_rnn[i])
            MDL.append(l_rnn[i])

        return out, x_history_stack, COMP_stack, MDL


class RNN_UAM(nn.Module):

    def __init__(self,num_classes=10, isTest=False):
        super(RNN_UAM, self).__init__()

        self.rnn = nn.RNNCell(         
            input_size=INPUT_SIZE,
            hidden_size=100,
            nonlinearity='tanh'
        )
        self.out = nn.Linear(100, num_classes)
        self.isTest = isTest 
        self.timestep = 5
        
    def RegularityEN(self, x, x_history, COMP):

        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)        
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP)))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        l_nml_x = COMP - sum_p_x_thetax
        # x = torch.mul(l_nml_x,x)
        
        return x, COMP, l_nml_x

    def cov(self, m, y=None):
        m = torch.t(m)
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def forward(self, x, x_history_stack=None,COMP_stack=None,inputRN=False, isTest=False):
        
        self.isTest = isTest
        
        x = x.view(-1,INPUT_SIZE)
        
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if inputRN:
            if x_history_stack==None:
                x_history_stats = x_stats
                COMP = torch.tensor(0).type(dtype)
            else:
                if self.isTest:
                    x_history_stats = x_history_stack[0] 
                else:
                    x_history_stats = [(x_history_stack[0][0]*x_history_stack[0][2]+x_stats[0]*x_stats[2])/(x_history_stack[0][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[0][2]-1)*torch.pow(x_history_stack[0][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[0][2]+x_stats[2]-2)),
                                   x_history_stack[0][2]+x_stats[2]]
                COMP = COMP_stack[0]
            x_history_input = x_history_stats
            x, COMP_input, l_input = self.RegularityEN(x,x_history_stats,COMP)
        else:
            COMP_input = -1
            x_history_input = -1
            l_input = -1
        COMP_input = COMP_input
        input_x = x
        
        COMP_rnn,l_rnn,x_history_rnn = [],[],[]
        for t in np.arange(self.timestep):
            if t == 0:
                x = self.rnn(input_x)
            else:
                x = self.rnn(input_x,x) 
            x_stats = [torch.mean(x), torch.std(x), torch.numel(x)]
            if x_history_stack==None:
                x_history_stats = x_stats
                COMP = torch.tensor(0).type(dtype)
            else:
                # if self.isTest:
                #     x_history_stats = x_history_stack[1+t] 
                # else:
                #     x_history_stats = [(x_history_stack[1+t][0]*x_history_stack[1+t][2]+x_stats[0]*x_stats[2])/(x_history_stack[1+t][2]+x_stats[2]),torch.sqrt(((x_history_stack[1+t][2]-1)*torch.pow(x_history_stack[1+t][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[1+t][2]+x_stats[2]-2)),
                #                    x_history_stack[1+t][2]+x_stats[2]]
                # COMP = COMP_stack[1+t]
                if self.isTest:
                    x_history_stats = x_history_stack[1] 
                else:
                    x_history_stats = [(x_history_stack[1][0]*x_history_stack[1][2]+x_stats[0]*x_stats[2])/(x_history_stack[1][2]+x_stats[2]),torch.sqrt(((x_history_stack[1][2]-1)*torch.pow(x_history_stack[1][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[1][2]+x_stats[2]-2)),
                                   x_history_stack[1][2]+x_stats[2]]
                COMP = COMP_stack[1]
                COMP_stack[1] = COMP
                x_history_stack[1] = x_history_stats
            
            x_history_rnn_t = x_history_stats
            COMP_rnn_t = -1
            x, COMP_rnn_t, l_rnn_t = self.RegularityEN(x,x_history_stats,COMP)
            COMP_rnn.append(COMP_rnn_t)
            l_rnn.append(l_rnn_t)
            x_history_rnn.append(x_history_rnn_t)

        x = self.out(x)
        out = F.log_softmax(x, dim=1)
        
        x_history_stack = [x_history_input]
        COMP_stack = [COMP_input]
        MDL = [l_input]
        for i in np.arange(self.timestep):
            x_history_stack.append(x_history_rnn[i])
            COMP_stack.append(COMP_rnn[i])
            MDL.append(l_rnn[i])

        return out, x_history_stack, COMP_stack, MDL


class NNRN_UAM(nn.Module):

    def __init__(self,num_classes=10, isTest=False):
        super(NNRN_UAM, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)
        self.isTest = isTest    
        
    def RegularityEN(self, x, x_history, COMP):

        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)        
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP)))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        l_nml_x = COMP - sum_p_x_thetax
        x = torch.mul(l_nml_x,x)
        
        return x, COMP, l_nml_x

    def cov(self, m, y=None):
        m = torch.t(m)
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def forward(self, x, x_history_stack=None,COMP_stack=None,inputRN=False, isTest=False):
        
        self.isTest = isTest
        
        x = x.view(-1,INPUT_SIZE)
        
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if inputRN:
            if x_history_stack==None:
                x_history_stats = x_stats
                COMP = torch.tensor(0).type(dtype)
            else:
                if self.isTest:
                    x_history_stats = x_history_stack[0] 
                else:
                    x_history_stats = [(x_history_stack[0][0]*x_history_stack[0][2]+x_stats[0]*x_stats[2])/(x_history_stack[0][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[0][2]-1)*torch.pow(x_history_stack[0][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[0][2]+x_stats[2]-2)),
                                   x_history_stack[0][2]+x_stats[2]]
                COMP = COMP_stack[0]
            x_history_input = x_history_stats
            x, COMP_input, l_input = self.RegularityEN(x,x_history_stats,COMP)
        else:
            COMP_input = -1
            x_history_input = -1
            l_input = -1
        COMP_input = COMP_input
        
        x = self.fc1(x)
        x_stats = [torch.mean(x), torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:
            if self.isTest:
                x_history_stats = x_history_stack[1] 
            else:
                x_history_stats = [(x_history_stack[1][0]*x_history_stack[1][2]+x_stats[0]*x_stats[2])/(x_history_stack[1][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[1][2]-1)*torch.pow(x_history_stack[1][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[1][2]+x_stats[2]-2)),
                                   x_history_stack[1][2]+x_stats[2]]
            COMP = COMP_stack[1]
        x_history_fc1 = x_history_stats
        COMP_fc1 = -1
        x, COMP_fc1, l_fc1 = self.RegularityEN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc2(x)
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:
            if self.isTest:
                x_history_stats = x_history_stack[2] 
            else:
                x_history_stats = [(x_history_stack[2][0]*x_history_stack[2][2]+x_stats[0]*x_stats[2])/(x_history_stack[2][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[2][2]-1)*torch.pow(x_history_stack[2][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[2][2]+x_stats[2]-2)),
                                   x_history_stack[2][2]+x_stats[2]]
            COMP = COMP_stack[2]
        x_history_fc2 = x_history_stats
        COMP_fc2 = -1
        x, COMP_fc2, l_fc2 = self.RegularityEN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        
        return out, x_history_stack, COMP_stack, [l_input, l_fc1, l_fc2]

class NN_UAM(nn.Module):

    def __init__(self,num_classes=10, isTest=False):
        super(NN_UAM, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)
        self.isTest = isTest    
        
    def RegularityEN(self, x, x_history, COMP):

        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)        
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP)))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        l_nml_x = COMP - sum_p_x_thetax
        # x = torch.mul(l_nml_x,x)
        
        return x, COMP, l_nml_x

    def cov(self, m, y=None):
        m = torch.t(m)
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def forward(self, x, x_history_stack=None,COMP_stack=None,inputRN=False, isTest=False):
        
        self.isTest = isTest
        
        x = x.view(-1,INPUT_SIZE)
        
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if inputRN:
            if x_history_stack==None:
                x_history_stats = x_stats
                COMP = torch.tensor(0).type(dtype)
            else:
                if self.isTest:
                    x_history_stats = x_history_stack[0] 
                else:
                    x_history_stats = [(x_history_stack[0][0]*x_history_stack[0][2]+x_stats[0]*x_stats[2])/(x_history_stack[0][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[0][2]-1)*torch.pow(x_history_stack[0][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[0][2]+x_stats[2]-2)),
                                   x_history_stack[0][2]+x_stats[2]]
                COMP = COMP_stack[0]
            x_history_input = x_history_stats
            x, COMP_input, l_input = self.RegularityEN(x,x_history_stats,COMP)
        else:
            COMP_input = -1
            x_history_input = -1
            l_input = -1
        COMP_input = COMP_input
        
        x = self.fc1(x)
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:
            if self.isTest:
                x_history_stats = x_history_stack[1] 
            else:
                x_history_stats = [(x_history_stack[1][0]*x_history_stack[1][2]+x_stats[0]*x_stats[2])/(x_history_stack[1][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[1][2]-1)*torch.pow(x_history_stack[1][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[1][2]+x_stats[2]-2)),
                                   x_history_stack[1][2]+x_stats[2]]
            COMP = COMP_stack[1]
        x_history_fc1 = x_history_stats
        COMP_fc1 = -1
        x, COMP_fc1, l_fc1 = self.RegularityEN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc2(x)
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:
            if self.isTest:
                x_history_stats = x_history_stack[2] 
            else:
                x_history_stats = [(x_history_stack[2][0]*x_history_stack[2][2]+x_stats[0]*x_stats[2])/(x_history_stack[2][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[2][2]-1)*torch.pow(x_history_stack[2][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[2][2]+x_stats[2]-2)),
                                   x_history_stack[2][2]+x_stats[2]]
            COMP = COMP_stack[2]
        x_history_fc2 = x_history_stats
        COMP_fc2 = -1
        x, COMP_fc2, l_fc2 = self.RegularityEN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        
        return out, x_history_stack, COMP_stack, [l_input, l_fc1, l_fc2]


class NNBN(nn.Module):
    def __init__(self,num_classes=10):
        super(NNBN, self).__init__()

        self.input_bn = nn.BatchNorm1d(INPUT_SIZE)
        self.fc1 = nn.Linear(INPUT_SIZE, 1000)
        self.fc1_bn = nn.BatchNorm1d(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_bn = nn.BatchNorm1d(1000)
        self.fc3 = nn.Linear(1000, num_classes)    

    def forward(self, x):
        
        x = x.view(-1,INPUT_SIZE)
        # x = self.input_bn(x)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)



class NNLN(nn.Module):
    def __init__(self,num_classes=10):
        super(NNLN, self).__init__()

        self.input_ln = nn.LayerNorm(INPUT_SIZE)
        self.fc1 = nn.Linear(INPUT_SIZE, 1000)
        self.fc1_ln = nn.LayerNorm(1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_ln = nn.LayerNorm(1000)
        self.fc3 = nn.Linear(1000, num_classes)    

    def forward(self, x):
        
        x = x.view(-1,INPUT_SIZE)
        # x = self.input_ln(x)
        x = self.fc1(x)
        x = self.fc1_ln(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.fc2_ln(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)



class NNWN(nn.Module):
    def __init__(self,num_classes=10):
        super(NNWN, self).__init__()

        self.fc1 = weightNorm(nn.Linear(INPUT_SIZE, 1000),name = "weight")
        self.fc2 = weightNorm(nn.Linear(1000, 1000),name = "weight")
        self.fc3 = weightNorm(nn.Linear(1000, num_classes),name = "weight")    

    def forward(self, x):
        
        x = x.view(-1,INPUT_SIZE)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)



class NNRN(nn.Module):
    def __init__(self,num_classes=10, isTest=False):
        super(NNRN, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)
        self.isTest = isTest    
        
    def RegularityEN(self, x, x_history, COMP):
        
#         p_thetax = torch.distributions.normal.Normal(torch.mean(x_history),torch.std(x_history))
#         p_x_thetax = torch.exp(p_thetax.log_prob(x))
#         COMP = torch.log2(torch.pow(2,COMP) + torch.sum(p_x_thetax))
#         p_nml_x = torch.div(p_x_thetax, torch.pow(2,COMP))
#         l_nml_x = -torch.log2(p_nml_x)
#         x = torch.mul(l_nml_x,x)

        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)        
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP)))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        l_nml_x = COMP - sum_p_x_thetax
        x = torch.mul(l_nml_x,x)
        
#         print(torch.sum(COMP), torch.sum(l_nml_x))
        return x, COMP

    def RegularityBN(self, x, x_history, COMP):
        
        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max),dim=1))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP),dim=1))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max),dim=1))
        l_nml_x = COMP - sum_p_x_thetax
        l_nml_x = l_nml_x.view(-1,1)
        x = torch.mul(l_nml_x.expand(x.shape[0],x.shape[1]),x)
        
#         p_x_history_thetax = torch.exp(p_thetax.log_prob(x_history))
#         COMP = torch.log2(torch.sum(p_x_history_thetax))
#         print(torch.mean(COMP), torch.mean(l_nml_x))
        return x, COMP

    def RegularityLN(self, x, x_history, COMP):
        
        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max),dim=0))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP),dim=0))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max),dim=0))
        l_nml_x = COMP - sum_p_x_thetax
        l_nml_x = l_nml_x.view(1,-1)
        x = torch.mul(l_nml_x.expand(x.shape[0],x.shape[1]),x)
        
#         print(torch.mean(COMP), torch.mean(l_nml_x))
        return x, COMP

    def cov(self, m, y=None):
        m = torch.t(m)
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def forward(self, x, x_history_stack=None,COMP_stack=None,inputRN=False, isTest=False):
        
        self.isTest = isTest
        
        x = x.view(-1,INPUT_SIZE)
        
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if inputRN:
            if x_history_stack==None:
                x_history_stats = x_stats
                COMP = torch.tensor(0).type(dtype)
            else:
                if self.isTest:
                    x_history_stats = x_history_stack[0] 
                else:
                    x_history_stats = [(x_history_stack[0][0]*x_history_stack[0][2]+x_stats[0]*x_stats[2])/(x_history_stack[0][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[0][2]-1)*torch.pow(x_history_stack[0][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[0][2]+x_stats[2]-2)),
                                   x_history_stack[0][2]+x_stats[2]]
                COMP = COMP_stack[0]
            x_history_input = x_history_stats
            x, COMP_input = self.RegularityEN(x,x_history_stats,COMP)
        else:
            COMP_input = -1
            x_history_input = -1
        COMP_input = COMP_input
        
        x = self.fc1(x)
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:
            if self.isTest:
                x_history_stats = x_history_stack[1] 
            else:
                x_history_stats = [(x_history_stack[1][0]*x_history_stack[1][2]+x_stats[0]*x_stats[2])/(x_history_stack[1][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[1][2]-1)*torch.pow(x_history_stack[1][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[1][2]+x_stats[2]-2)),
                                   x_history_stack[1][2]+x_stats[2]]
#             x_history = torch.cat((x_history_stack[0],x),0)
            COMP = COMP_stack[1]
        x_history_fc1 = x_history_stats
        COMP_fc1 = -1
        x, COMP_fc1 = self.RegularityEN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc2(x)
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:
            if self.isTest:
                x_history_stats = x_history_stack[2] 
            else:
                x_history_stats = [(x_history_stack[2][0]*x_history_stack[2][2]+x_stats[0]*x_stats[2])/(x_history_stack[2][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[2][2]-1)*torch.pow(x_history_stack[2][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[2][2]+x_stats[2]-2)),
                                   x_history_stack[2][2]+x_stats[2]]
#             x_history = torch.cat((x_history_stack[0],x),0)
            COMP = COMP_stack[2]
        x_history_fc2 = x_history_stats
        COMP_fc2 = -1
        x, COMP_fc2 = self.RegularityEN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        
        return out, x_history_stack, COMP_stack



class NNRNLN(nn.Module):
    def __init__(self,num_classes=10, isTest=False):
        super(NNRNLN, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)
        self.isTest = isTest    
        
        self.input_ln = nn.LayerNorm(INPUT_SIZE)
        self.fc1_ln = nn.LayerNorm(1000)
        self.fc2_ln = nn.LayerNorm(1000)   
        
    def RegularityEN(self, x, x_history, COMP):
        
        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)        
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP)))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        l_nml_x = COMP - sum_p_x_thetax
        x = torch.mul(l_nml_x,x)
        
#         print(torch.sum(COMP), torch.sum(l_nml_x))
        return x, COMP

    def forward(self, x, x_history_stack=None,COMP_stack=None,inputRN=False, isTest=False):
        
        self.isTest = isTest
        
        x = x.view(-1,INPUT_SIZE)
        # x = self.input_ln(x)
        
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if inputRN:
            if x_history_stack==None:
                x_history_stats = x_stats
                COMP = torch.tensor(0).type(dtype)
            else:
                if self.isTest:
                    x_history_stats = x_history_stack[0] 
                else:
                    x_history_stats = [(x_history_stack[0][0]*x_history_stack[0][2]+x_stats[0]*x_stats[2])/(x_history_stack[0][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[0][2]-1)*torch.pow(x_history_stack[0][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[0][2]+x_stats[2]-2)),
                                   x_history_stack[0][2]+x_stats[2]]
                COMP = COMP_stack[0]
            x_history_input = x_history_stats
            x, COMP_input = self.RegularityEN(x,x_history_stats,COMP)
        else:
            COMP_input = -1
            x_history_input = -1
        COMP_input = COMP_input
         
        x = self.fc1(x)
        x = self.fc1_ln(x)
        
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:
            if self.isTest:
                x_history_stats = x_history_stack[1] 
            else:
                x_history_stats = [(x_history_stack[1][0]*x_history_stack[1][2]+x_stats[0]*x_stats[2])/(x_history_stack[1][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[1][2]-1)*torch.pow(x_history_stack[1][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[1][2]+x_stats[2]-2)),
                                   x_history_stack[1][2]+x_stats[2]]
#             x_history = torch.cat((x_history_stack[0],x),0)
            COMP = COMP_stack[1]
        x_history_fc1 = x_history_stats
        COMP_fc1 = -1
        x, COMP_fc1 = self.RegularityEN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc2(x)
        x = self.fc2_ln(x)
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:
            if self.isTest:
                x_history_stats = x_history_stack[2] 
            else:
                x_history_stats = [(x_history_stack[2][0]*x_history_stack[2][2]+x_stats[0]*x_stats[2])/(x_history_stack[2][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[2][2]-1)*torch.pow(x_history_stack[2][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[2][2]+x_stats[2]-2)),
                                   x_history_stack[2][2]+x_stats[2]]
#             x_history = torch.cat((x_history_stack[0],x),0)
            COMP = COMP_stack[2]
        x_history_fc2 = x_history_stats
        COMP_fc2 = -1
        x, COMP_fc2 = self.RegularityEN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        
        return out, x_history_stack, COMP_stack



class NNRBN(nn.Module):
    def __init__(self,num_classes=10, isTest=False):
        super(NNRBN, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)
        self.isTest = isTest    


    def RegularityBN(self, x, x_history, COMP):
        
        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max),dim=1))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP),dim=1))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max),dim=1))
        l_nml_x = COMP - sum_p_x_thetax
        l_nml_x = l_nml_x.view(-1,1)
        x = torch.mul(l_nml_x.expand(x.shape[0],x.shape[1]),x)
        
        return x, COMP


    def forward(self, x, x_history_stack=None,COMP_stack=None,inputRN=False, isTest=False):
        
        self.isTest = isTest
        
        x = x.view(-1,INPUT_SIZE)
        
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if inputRN:
            if x_history_stack==None:
                x_history_stats = x_stats
                COMP = torch.tensor(0).type(dtype)
            else:
                if self.isTest:
                    x_history_stats = x_history_stack[0] 
                else:
                    x_history_stats = [(x_history_stack[0][0]*x_history_stack[0][2]+x_stats[0]*x_stats[2])/(x_history_stack[0][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[0][2]-1)*torch.pow(x_history_stack[0][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[0][2]+x_stats[2]-2)),
                                   x_history_stack[0][2]+x_stats[2]]
                COMP = COMP_stack[0]
            x_history_input = x_history_stats
            x, COMP_input = self.RegularityBN(x,x_history_stats,COMP)
        else:
            COMP_input = -1
            x_history_input = -1
        COMP_input = COMP_input
        
        x = self.fc1(x)
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:
            if self.isTest:
                x_history_stats = x_history_stack[1] 
            else:
                x_history_stats = [(x_history_stack[1][0]*x_history_stack[1][2]+x_stats[0]*x_stats[2])/(x_history_stack[1][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[1][2]-1)*torch.pow(x_history_stack[1][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[1][2]+x_stats[2]-2)),
                                   x_history_stack[1][2]+x_stats[2]]
#             x_history = torch.cat((x_history_stack[0],x),0)
            COMP = COMP_stack[1]
        x_history_fc1 = x_history_stats
        COMP_fc1 = -1
        x, COMP_fc1 = self.RegularityBN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc2(x)
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:
            if self.isTest:
                x_history_stats = x_history_stack[2] 
            else:
                x_history_stats = [(x_history_stack[2][0]*x_history_stack[2][2]+x_stats[0]*x_stats[2])/(x_history_stack[2][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[2][2]-1)*torch.pow(x_history_stack[2][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[2][2]+x_stats[2]-2)),
                                   x_history_stack[2][2]+x_stats[2]]
#             x_history = torch.cat((x_history_stack[0],x),0)
            COMP = COMP_stack[2]
        x_history_fc2 = x_history_stats
        COMP_fc2 = -1
        x, COMP_fc2 = self.RegularityBN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        
        return out, x_history_stack, COMP_stack



class NNRLN(nn.Module):
    def __init__(self,num_classes=10, isTest=False):
        super(NNRLN, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)
        self.isTest = isTest    

    def RegularityLN(self, x, x_history, COMP):
        
        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max),dim=0))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP),dim=0))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max),dim=0))
        l_nml_x = COMP - sum_p_x_thetax
        l_nml_x = l_nml_x.view(1,-1)
        x = torch.mul(l_nml_x.expand(x.shape[0],x.shape[1]),x)
        
#         print(torch.mean(COMP), torch.mean(l_nml_x))
        return x, COMP


    def forward(self, x, x_history_stack=None,COMP_stack=None,inputRN=False, isTest=False):
        
        self.isTest = isTest
        
        x = x.view(-1,INPUT_SIZE)
        
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if inputRN:
            if x_history_stack==None:
                x_history_stats = x_stats
                COMP = torch.tensor(0).type(dtype)
            else:
                if self.isTest:
                    x_history_stats = x_history_stack[0] 
                else:
                    x_history_stats = [(x_history_stack[0][0]*x_history_stack[0][2]+x_stats[0]*x_stats[2])/(x_history_stack[0][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[0][2]-1)*torch.pow(x_history_stack[0][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[0][2]+x_stats[2]-2)),
                                   x_history_stack[0][2]+x_stats[2]]
                COMP = COMP_stack[0]
            x_history_input = x_history_stats
            x, COMP_input = self.RegularityLN(x,x_history_stats,COMP)
        else:
            COMP_input = -1
            x_history_input = -1
        COMP_input = COMP_input
        
        x = self.fc1(x)
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:
            if self.isTest:
                x_history_stats = x_history_stack[1] 
            else:
                x_history_stats = [(x_history_stack[1][0]*x_history_stack[1][2]+x_stats[0]*x_stats[2])/(x_history_stack[1][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[1][2]-1)*torch.pow(x_history_stack[1][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[1][2]+x_stats[2]-2)),
                                   x_history_stack[1][2]+x_stats[2]]
#             x_history = torch.cat((x_history_stack[0],x),0)
            COMP = COMP_stack[1]
        x_history_fc1 = x_history_stats
        COMP_fc1 = -1
        x, COMP_fc1 = self.RegularityLN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc2(x)
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:
            if self.isTest:
                x_history_stats = x_history_stack[2] 
            else:
                x_history_stats = [(x_history_stack[2][0]*x_history_stack[2][2]+x_stats[0]*x_stats[2])/(x_history_stack[2][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[2][2]-1)*torch.pow(x_history_stack[2][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[2][2]+x_stats[2]-2)),
                                   x_history_stack[2][2]+x_stats[2]]
#             x_history = torch.cat((x_history_stack[0],x),0)
            COMP = COMP_stack[2]
        x_history_fc2 = x_history_stats
        COMP_fc2 = -1
        x, COMP_fc2 = self.RegularityLN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        
        return out, x_history_stack, COMP_stack



class NNSN(nn.Module):
    def __init__(self,num_classes=10, isTest=False):
        super(NNSN, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)   
        self.num_classes = num_classes
        self.isTest = isTest
    
    def RegularityEN(self, x, x_history, COMP, s):

        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)        
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP)))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
#         print(COMP, sum_p_x_thetax, torch.log(s).shape, x.shape)
        l_nml_x = COMP - sum_p_x_thetax - torch.log(s)
        x = torch.mul(l_nml_x,x)
        return x, COMP

    def RegularityBN(self, x, x_history, COMP, s):
        
        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max),dim=1))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP),dim=1))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max),dim=1))
        l_nml_x = COMP - sum_p_x_thetax - torch.log(s,dim=1)
        l_nml_x = l_nml_x.view(-1,1)
        x = torch.mul(l_nml_x.expand(x.shape[0],x.shape[1]),x)
        return x, COMP

    def RegularityLN(self, x, x_history, COMP, s):
        
        p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
        log_p_x_thetax = p_thetax.log_prob(x)
        log_p_x_max = torch.max(log_p_x_thetax)
        sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max),dim=0))
        if not self.isTest:
            COMP_max = torch.max(COMP)
            if COMP_max > log_p_x_max:
                COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP),dim=0))
            else:
                COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max),dim=0))
        l_nml_x = COMP - sum_p_x_thetax - torch.log(s,dim=0)
        l_nml_x = l_nml_x.view(1,-1)
        x = torch.mul(l_nml_x.expand(x.shape[0],x.shape[1]),x)
        return x, COMP

    def cov(self, m, y=None):
        m = torch.t(m)
        if y is not None:
            m = torch.cat((m, y), dim=0)
        m_exp = torch.mean(m, dim=1)
        x = m - m_exp[:, None]
        cov = 1 / (x.size(1) - 1) * x.mm(x.t())
        return cov

    def forward(self, x, y=None, x_history_stack=None,COMP_stack=None,inputRN=False, isTest=False):
        
        self.isTest = isTest
        x = x.view(-1,INPUT_SIZE)
        
        s = torch.ones((x.shape)).type(dtype)
        if y is not None:
            for label in np.arange(self.num_classes):
                s[y==label] = len((y==label).nonzero()) / len(y)
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if inputRN:
            if x_history_stack==None:
                x_history_stats = x_stats
                COMP = torch.tensor(0).type(dtype)
            else:
                if self.isTest:
                    x_history_stats = x_history_stack[0] 
                else:
                    x_history_stats = [(x_history_stack[0][0]*x_history_stack[0][2]+x_stats[0]*x_stats[2])/(x_history_stack[0][2]+x_stats[2]),
                                       torch.sqrt(((x_history_stack[0][2]-1)*torch.pow(x_history_stack[0][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[0][2]+x_stats[2]-2)),
                                       x_history_stack[0][2]+x_stats[2]]
                COMP = COMP_stack[0]
            x_history_input = x_history_stats
            x, COMP_input = self.RegularityEN(x,x_history_stats,COMP,s)
        else:
            COMP_input = -1
            x_history_input = -1
        COMP_input = COMP_input
        
        x = self.fc1(x)
        
        s = torch.ones((x.shape)).type(dtype)
        if y is not None:
            for label in np.arange(self.num_classes):
                s[y==label] = len((y==label).nonzero()) / len(y)
                
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:          
            if self.isTest:
                x_history_stats = x_history_stack[1]             
            else:
                x_history_stats = [(x_history_stack[1][0]*x_history_stack[1][2]+x_stats[0]*x_stats[2])/(x_history_stack[1][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[1][2]-1)*torch.pow(x_history_stack[1][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[1][2]+x_stats[2]-2)),
                                   x_history_stack[1][2]+x_stats[2]]
#             x_history = torch.cat((x_history_stack[0],x),0)
            COMP = COMP_stack[1]
        x_history_fc1 = x_history_stats
        COMP_fc1 = -1
        x, COMP_fc1 = self.RegularityEN(x,x_history_stats,COMP,s)
        x = F.relu(x)
        
        x = self.fc2(x)
        
        s = torch.ones((x.shape)).type(dtype)
        if y is not None:
            for label in np.arange(self.num_classes):
                s[y==label] = len((y==label).nonzero()) / len(y)
                
        x_stats = [torch.mean(x),torch.std(x), torch.numel(x)]
        if x_history_stack==None:
            x_history_stats = x_stats
            COMP = torch.tensor(0).type(dtype)
        else:          
            if self.isTest:
                x_history_stats = x_history_stack[2] 
            else:
                x_history_stats = [(x_history_stack[2][0]*x_history_stack[2][2]+x_stats[0]*x_stats[2])/(x_history_stack[2][2]+x_stats[2]),
                                   torch.sqrt(((x_history_stack[2][2]-1)*torch.pow(x_history_stack[2][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[2][2]+x_stats[2]-2)),
                                   x_history_stack[2][2]+x_stats[2]]
#             x_history = torch.cat((x_history_stack[0],x),0)
            COMP = COMP_stack[2]
        x_history_fc2 = x_history_stats
        COMP_fc2 = -1
        x, COMP_fc2 = self.RegularityEN(x,x_history_stats,COMP,s)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = F.log_softmax(x, dim=1)
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        
        return out, x_history_stack, COMP_stack


class NN(nn.Module):
    def __init__(self,num_classes=10):
        super(NN, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, num_classes)

    def forward(self, x):
        
        x = x.view(-1,INPUT_SIZE)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return F.log_softmax(x, dim=1)


class CNN(nn.Module):
    def __init__(self,num_classes=10):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

class RNN(nn.Module):
    def __init__(self,rnn_h_size=64,num_classes=10):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=rnn_h_size,         # rnn hidden unit
            num_layers=1,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.out = nn.Linear(rnn_h_size, num_classes)

    def forward(self, x):
        
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
#         r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        
#         x = np.repeat(x,TIME_STEP,axis=0).view(-1, TIME_STEP, INPUT_SIZE)
        x = x.view(-1, 1, INPUT_SIZE)
        for t in np.arange(TIME_STEP):
            r_out, h = self.rnn(x, None)   # None represents zero initial hidden state
        
        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out, r_out


