##########################
# Modified from https://github.com/lcswillems/rl-starter-files

# by Baihan Lin for Unsupervised Attention Mechanism
# https://arxiv.org/abs/1902.10658

# Environment: Python 3.6+, Pytorch 1.0
##########################

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac

import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def time_usage(func):
    def wrapper(*args, **kwargs):
        beg_ts = time.time()
        retval = func(*args, **kwargs)
        end_ts = time.time()
        print("elapsed time: %f" % (end_ts - beg_ts))
        return retval
    return wrapper

def getMDL(x, x_history, COMP, RN=False, s=None):
    p_thetax = torch.distributions.normal.Normal(x_history[0],x_history[1])
    log_p_x_thetax = p_thetax.log_prob(x)        
    log_p_x_max = torch.max(log_p_x_thetax)
    sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
    COMP_max = torch.max(COMP)
    if COMP_max > log_p_x_max:
        COMP = COMP + torch.log(1+torch.sum(torch.exp(log_p_x_thetax - COMP)))
    else:
        COMP = log_p_x_max + torch.log(torch.exp(COMP - log_p_x_max)+torch.sum(torch.exp(log_p_x_thetax - log_p_x_max)))
    if s is None:
        l_nml_x = COMP - sum_p_x_thetax
    else:
        l_nml_x = COMP - sum_p_x_thetax - torch.log(s)
    if RN: x = torch.mul(l_nml_x,x)
    return x, COMP, l_nml_x

def getHistoryStats(x,x_history_stack,COMP_stack,l,isTest=False):        
    x_stats = [torch.mean(x),torch.std(x),torch.numel(x)]
    if x_history_stack==None:
        x_history_stats = x_stats
        COMP = torch.tensor(0.0).to(device)
    else:
        if isTest:
            x_history_stats = x_history_stack[l] 
        else:
            x_history_stats = [(x_history_stack[l][0]*x_history_stack[l][2]+x_stats[0]*x_stats[2])/(x_history_stack[l][2]+x_stats[2]),torch.sqrt(((x_history_stack[l][2]-1)*torch.pow(x_history_stack[l][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[l][2]+x_stats[2]-2)),x_history_stack[l][2]+x_stats[2]]
        COMP = COMP_stack[l]
    return x_history_stats, COMP

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

class ACModelLN(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        self.Tanh = nn.Tanh()
        
        # Define actor's model
        self.actor_fc1 = nn.Linear(self.embedding_size, 64)
        self.actor_fc1_ln = nn.LayerNorm(64)
        self.actor_fc2 = nn.Linear(64, action_space.n)
        self.actor_fc2_ln = nn.LayerNorm(action_space.n)

        # Define critic's model
        self.critic_fc1 = nn.Linear(self.embedding_size, 64)
        self.critic_fc1_ln = nn.LayerNorm(64)
        self.critic_fc2 = nn.Linear(64, 1)
        self.critic_fc2_ln = nn.LayerNorm(1)

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor_fc1(embedding)
        x = self.actor_fc1_ln(x)
        x = self.Tanh(x)
        x = self.actor_fc2(x)
        x = self.actor_fc2_ln(x)

        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic_fc1(embedding)
        x = self.critic_fc1_ln(x)
        x = self.Tanh(x)
        x = self.critic_fc2(x)
        x = self.critic_fc2_ln(x)
        
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


class ACModelRN(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        self.Tanh = nn.Tanh()
        
        # Define actor's model
        self.actor_fc1 = nn.Linear(self.embedding_size, 64)
        self.actor_fc2 = nn.Linear(64, action_space.n)

        # Define critic's model
        self.critic_fc1 = nn.Linear(self.embedding_size, 64)
        self.critic_fc2 = nn.Linear(64, 1)

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory,x_history_stack=None,COMP_stack=None,RN=True):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        x_history_new,COMP_new,l_new = [],[],[]

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor_fc1(embedding)

        x_history_stats,COMP = getHistoryStats(x,x_history_stack,COMP_stack,0)       
        x, COMP, l = getMDL(x,x_history_stats,COMP,RN)
        COMP_new.append(COMP)
        l_new.append(l)
        x_history_new.append(x_history_stats)

        x = self.Tanh(x)
        x = self.actor_fc2(x)

        x_history_stats,COMP = getHistoryStats(x,x_history_stack,COMP_stack,1)       
        x, COMP, l = getMDL(x,x_history_stats,COMP,RN)
        COMP_new.append(COMP)
        l_new.append(l)
        x_history_new.append(x_history_stats)

        if torch.sum(x != x) > 0: x = torch.ones((x.shape))/sum(torch.ones((x.shape)))
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic_fc1(embedding)
        
        x_history_stats,COMP = getHistoryStats(x,x_history_stack,COMP_stack,2)       
        x, COMP, l = getMDL(x,x_history_stats,COMP,RN)
        COMP_new.append(COMP)
        l_new.append(l)
        x_history_new.append(x_history_stats)   
        
        x = self.Tanh(x)
        x = self.critic_fc2(x)
        
        x_history_stats,COMP = getHistoryStats(x,x_history_stack,COMP_stack,3)       
        x, COMP, l = getMDL(x,x_history_stats,COMP,RN)
        COMP_new.append(COMP)
        l_new.append(l)
        x_history_new.append(x_history_stats) 
        
        value = x.squeeze(1)

        return dist, value, memory, x_history_new,COMP_new,l_new

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]



class ACModelRNLN(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        self.Tanh = nn.Tanh()
        
        # Define actor's model
        self.actor_fc1 = nn.Linear(self.embedding_size, 64)
        self.actor_fc1_ln = nn.LayerNorm(64)
        self.actor_fc2 = nn.Linear(64, action_space.n)
        self.actor_fc2_ln = nn.LayerNorm(action_space.n)

        # Define critic's model
        self.critic_fc1 = nn.Linear(self.embedding_size, 64)
        self.critic_fc1_ln = nn.LayerNorm(64)
        self.critic_fc2 = nn.Linear(64, 1)
        self.critic_fc2_ln = nn.LayerNorm(1)

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory,x_history_stack=None,COMP_stack=None,RN=True):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        x_history_new,COMP_new,l_new = [],[],[]

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        x = self.actor_fc1(embedding)

        x_history_stats,COMP = getHistoryStats(x,x_history_stack,COMP_stack,0)       
        x, COMP, l = getMDL(x,x_history_stats,COMP,RN)
        COMP_new.append(COMP)
        l_new.append(l)
        x_history_new.append(x_history_stats)

        x = self.actor_fc1_ln(x)
        x = self.Tanh(x)
        x = self.actor_fc2(x)

        x_history_stats,COMP = getHistoryStats(x,x_history_stack,COMP_stack,1)       
        x, COMP, l = getMDL(x,x_history_stats,COMP,RN)
        COMP_new.append(COMP)
        l_new.append(l)
        x_history_new.append(x_history_stats)

        x = self.actor_fc2_ln(x)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        x = self.critic_fc1(embedding)
        
        x_history_stats,COMP = getHistoryStats(x,x_history_stack,COMP_stack,2)       
        x, COMP, l = getMDL(x,x_history_stats,COMP,RN)
        COMP_new.append(COMP)
        l_new.append(l)
        x_history_new.append(x_history_stats)   
        
        x = self.critic_fc1_ln(x)
        x = self.Tanh(x)
        x = self.critic_fc2(x)
        
        x_history_stats,COMP = getHistoryStats(x,x_history_stack,COMP_stack,3)       
        x, COMP, l = getMDL(x,x_history_stats,COMP,RN)
        COMP_new.append(COMP)
        l_new.append(l)
        x_history_new.append(x_history_stats) 
        
        x = self.critic_fc2_ln(x)

        value = x.squeeze(1)

        return dist, value, memory, x_history_new,COMP_new,l_new

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]


