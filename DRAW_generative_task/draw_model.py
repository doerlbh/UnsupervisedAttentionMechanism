##########################
# Modified from https://github.com/chenzhaomin123/draw_pytorch

# by Baihan Lin for Unsupervised Attention Mechanism
# https://arxiv.org/abs/1902.10658

# Environment: Python 3.6+, Pytorch 1.0
##########################

import torch
import torch.nn as nn
from utility import *
import torch.functional as F

import time

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
        COMP = torch.tensor(0.0)
    else:
        if isTest:
            x_history_stats = x_history_stack[l] 
        else:
            x_history_stats = [(x_history_stack[l][0]*x_history_stack[l][2]+x_stats[0]*x_stats[2])/(x_history_stack[l][2]+x_stats[2]),torch.sqrt(((x_history_stack[l][2]-1)*torch.pow(x_history_stack[l][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[l][2]+x_stats[2]-2)),x_history_stack[l][2]+x_stats[2]]
        COMP = COMP_stack[l]
    return x_history_stats, COMP

class DrawModel(nn.Module):
    def __init__(self,T,A,B,z_size,N,dec_size,enc_size):
        super(DrawModel,self).__init__()
        self.T = T
        # self.batch_size = batch_size
        self.A = A
        self.B = B
        self.z_size = z_size
        self.N = N
        self.dec_size = dec_size
        self.enc_size = enc_size
        self.cs = [0] * T
        self.logsigmas,self.sigmas,self.mus = [0] * T,[0] * T,[0] * T

        self.encoder = nn.LSTMCell(2 * N * N + dec_size, enc_size)
        self.encoder_gru = nn.GRUCell(2 * N * N + dec_size, enc_size)
        self.mu_linear = nn.Linear(dec_size, z_size)
        self.sigma_linear = nn.Linear(dec_size, z_size)

        self.decoder = nn.LSTMCell(z_size,dec_size)
        self.decoder_gru = nn.GRUCell(z_size,dec_size)
        self.dec_linear = nn.Linear(dec_size,5)
        self.dec_w_linear = nn.Linear(dec_size,N*N)

        self.sigmoid = nn.Sigmoid()

    def normalSample(self):
        return Variable(torch.randn(self.batch_size,self.z_size))

    # correct
    def compute_mu(self,g,rng,delta):
        rng_t,delta_t = align(rng,delta)
        tmp = (rng_t - self.N / 2 - 0.5) * delta_t
        tmp_t,g_t = align(tmp,g)
        mu = tmp_t + g_t
        return mu

    # correct
    def filterbank(self,gx,gy,sigma2,delta):
        rng = Variable(torch.arange(0,self.N).view(1,-1))
        mu_x = self.compute_mu(gx,rng,delta)
        mu_y = self.compute_mu(gy,rng,delta)

        a = Variable(torch.arange(0,self.A).view(1,1,-1))
        b = Variable(torch.arange(0,self.B).view(1,1,-1))

        mu_x = mu_x.view(-1,self.N,1)
        mu_y = mu_y.view(-1,self.N,1)
        sigma2 = sigma2.view(-1,1,1)

        Fx = self.filterbank_matrices(a,mu_x,sigma2)
        Fy = self.filterbank_matrices(b,mu_y,sigma2)

        return Fx,Fy

    def forward(self,x):
        self.batch_size = x.size()[0]
        h_dec_prev = Variable(torch.zeros(self.batch_size,self.dec_size))
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))

        enc_state = Variable(torch.zeros(self.batch_size,self.enc_size))
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))
        for t in range(self.T):
            c_prev = Variable(torch.zeros(self.batch_size,self.A * self.B)) if t == 0 else self.cs[t-1]
            x_hat = x - self.sigmoid(c_prev)     # 3
            r_t = self.read(x,x_hat,h_dec_prev)
            h_enc_prev,enc_state = self.encoder(torch.cat((r_t,h_dec_prev),1),(h_enc_prev,enc_state))
            # h_enc = self.encoder_gru(torch.cat((r_t,h_dec_prev),1),h_enc_prev)
            z,self.mus[t],self.logsigmas[t],self.sigmas[t] = self.sampleQ(h_enc_prev)
            h_dec,dec_state = self.decoder(z, (h_dec_prev, dec_state))
            # h_dec = self.decoder_gru(z, h_dec_prev)
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec

    def loss(self,x,**kwargs):
        self.forward(x,**kwargs)
        criterion = nn.BCELoss()
        x_recons = self.sigmoid(self.cs[-1])
        Lx = criterion(x_recons,x) * self.A * self.B
        Lz = 0
        kl_terms = [0] * T
        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]
            # Lz += (0.5 * (mu_2 + sigma_2 - 2 * logsigma))    # 11
            kl_terms[t] = 0.5 * torch.sum(mu_2+sigma_2-2 * logsigma,1) - self.T * 0.5
            Lz += kl_terms[t]
        # Lz -= self.T / 2
        Lz = torch.mean(Lz)    ####################################################
        loss = Lz + Lx    # 12
        return loss

    # correct
    def filterbank_matrices(self,a,mu_x,sigma2,epsilon=1e-9):
        t_a,t_mu_x = align(a,mu_x)
        temp = t_a - t_mu_x
        temp,t_sigma = align(temp,sigma2)
        temp = temp / (t_sigma * 2)
        F = torch.exp(-torch.pow(temp,2))
        F = F / (F.sum(2,True).expand_as(F) + epsilon)
        return F

    #correct
    def attn_window(self,h_dec):
        params = self.dec_linear(h_dec)
        gx_,gy_,log_sigma_2,log_delta,log_gamma = params.split(1,1)  #21

        # gx_ = Variable(torch.ones(4,1))
        # gy_ = Variable(torch.ones(4, 1) * 2)
        # log_sigma_2 = Variable(torch.ones(4, 1) * 3)
        # log_delta = Variable(torch.ones(4, 1) * 4)
        # log_gamma = Variable(torch.ones(4, 1) * 5)

        gx = (self.A + 1) / 2 * (gx_ + 1)    # 22
        gy = (self.B + 1) / 2 * (gy_ + 1)    # 23
        delta = (max(self.A,self.B) - 1) / (self.N - 1) * torch.exp(log_delta)  # 24
        sigma2 = torch.exp(log_sigma_2)
        gamma = torch.exp(log_gamma)

        return self.filterbank(gx,gy,sigma2,delta),gamma
    # correct
    def read(self,x,x_hat,h_dec_prev):
        (Fx,Fy),gamma = self.attn_window(h_dec_prev)
        def filter_img(img,Fx,Fy,gamma,A,B,N):
            Fxt = Fx.transpose(2,1)
            img = img.view(-1,B,A)
            # img = img.transpose(2,1)
            # glimpse = matmul(Fy,matmul(img,Fxt))
            glimpse = Fy.bmm(img.bmm(Fxt))
            glimpse = glimpse.view(-1,N*N)
            return glimpse * gamma.view(-1,1).expand_as(glimpse)
        x = filter_img(x,Fx,Fy,gamma,self.A,self.B,self.N)
        x_hat = filter_img(x_hat,Fx,Fy,gamma,self.A,self.B,self.N)
        return torch.cat((x,x_hat),1)

    # correct
    def write(self,h_dec=0):
        w = self.dec_w_linear(h_dec)
        w = w.view(self.batch_size,self.N,self.N)
        # w = Variable(torch.ones(4,5,5) * 3)
        # self.batch_size = 4
        (Fx,Fy),gamma = self.attn_window(h_dec)
        Fyt = Fy.transpose(2,1)
        # wr = matmul(Fyt,matmul(w,Fx))
        wr = Fyt.bmm(w.bmm(Fx))
        wr = wr.view(self.batch_size,self.A*self.B)
        return wr / gamma.view(-1,1).expand_as(wr)

    def sampleQ(self,h_enc):
        e = self.normalSample()
        # mu_sigma = self.mu_sigma_linear(h_enc)
        # mu = mu_sigma[:, :self.z_size]
        # log_sigma = mu_sigma[:, self.z_size:]
        mu = self.mu_linear(h_enc)           # 1
        log_sigma = self.sigma_linear(h_enc) # 2
        sigma = torch.exp(log_sigma)

        return mu + sigma * e , mu , log_sigma, sigma

    def generate(self,batch_size=64):
        self.batch_size = batch_size
        h_dec_prev = Variable(torch.zeros(self.batch_size,self.dec_size),volatile = True)
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size),volatile = True)

        for t in range(self.T):
            c_prev = Variable(torch.zeros(self.batch_size, self.A * self.B)) if t == 0 else self.cs[t - 1]
            z = self.normalSample()
            h_dec, dec_state = self.decoder(z, (h_dec_prev, dec_state))
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec
        imgs = []
        for img in self.cs:
            imgs.append(self.sigmoid(img).cpu().data.numpy())
        return imgs

class DrawModelLN(DrawModel):
    def __init__(self,T,A,B,z_size,N,dec_size,enc_size):
        super(DrawModelLN,self).__init__(T,A,B,z_size,N,dec_size,enc_size)
        self.enc_ln = nn.LayerNorm(enc_size)
        self.dec_ln = nn.LayerNorm(dec_size)

    def forward(self,x):
        self.batch_size = x.size()[0]
        h_dec_prev = Variable(torch.zeros(self.batch_size,self.dec_size))
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))

        enc_state = Variable(torch.zeros(self.batch_size,self.enc_size))
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))
        for t in range(self.T):
            c_prev = Variable(torch.zeros(self.batch_size,self.A * self.B)) if t == 0 else self.cs[t-1]
            x_hat = x - self.sigmoid(c_prev)     # 3
            r_t = self.read(x,x_hat,h_dec_prev)
            h_enc_prev,enc_state = self.encoder(torch.cat((r_t,h_dec_prev),1),(h_enc_prev,enc_state))
            h_enc_prev = self.enc_ln(h_enc_prev)
            # h_enc = self.encoder_gru(torch.cat((r_t,h_dec_prev),1),h_enc_prev)
            z,self.mus[t],self.logsigmas[t],self.sigmas[t] = self.sampleQ(h_enc_prev)
            h_dec,dec_state = self.decoder(z, (h_dec_prev, dec_state))
            h_dec = self.dec_ln(h_dec)
            # h_dec = self.decoder_gru(z, h_dec_prev)
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec

class DrawModelRN(DrawModel):
    def __init__(self,T,A,B,z_size,N,dec_size,enc_size):
        super(DrawModelRN,self).__init__(T,A,B,z_size,N,dec_size,enc_size)
        # self.enc_ln = nn.LayerNorm(enc_size)
        # self.dec_ln = nn.LayerNorm(dec_size)

    def forward(self,x,x_history_stack=None,COMP_stack=None):
        self.batch_size = x.size()[0]
        h_dec_prev = Variable(torch.zeros(self.batch_size,self.dec_size))
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))

        x_history_new,COMP_new,l_new = [],[],[]

        enc_state = Variable(torch.zeros(self.batch_size,self.enc_size))
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))
        for t in range(self.T):
            c_prev = Variable(torch.zeros(self.batch_size,self.A * self.B)) if t == 0 else self.cs[t-1]
            x_hat = x - self.sigmoid(c_prev)     # 3
            r_t = self.read(x,x_hat,h_dec_prev)
            h_enc_prev,enc_state = self.encoder(torch.cat((r_t,h_dec_prev),1),(h_enc_prev,enc_state))
            # h_enc_prev = self.enc_ln(h_enc_prev)
            
            x_history_stats,COMP = getHistoryStats(h_enc_prev,x_history_stack,COMP_stack,t*2)       
            h_enc_prev, COMP, l = getMDL(h_enc_prev,x_history_stats,COMP,True)
            COMP_new.append(COMP)
            l_new.append(l)
            x_history_new.append(x_history_stats)

            # h_enc = self.encoder_gru(torch.cat((r_t,h_dec_prev),1),h_enc_prev)
            z,self.mus[t],self.logsigmas[t],self.sigmas[t] = self.sampleQ(h_enc_prev)
            h_dec,dec_state = self.decoder(z, (h_dec_prev, dec_state))
            # h_dec = self.dec_ln(h_dec)

            x_history_stats,COMP = getHistoryStats(h_dec,x_history_stack,COMP_stack,t*2+1)       
            h_dec,COMP,l = getMDL(h_dec,x_history_stats,COMP,True)
            COMP_new.append(COMP)
            l_new.append(l)
            x_history_new.append(x_history_stats)

            # h_dec = self.decoder_gru(z, h_dec_prev)
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec
        
        return x_history_new,COMP_new,l_new

    def loss(self,x,x_history_stack=None,COMP_stack=None):
        x_history_new,COMP_new,l_new = self.forward(x,x_history_stack,COMP_stack)
        criterion = nn.BCELoss()
        x_recons = self.sigmoid(self.cs[-1])
        Lx = criterion(x_recons,x) * self.A * self.B
        Lz = 0
        kl_terms = [0] * T
        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]
            # Lz += (0.5 * (mu_2 + sigma_2 - 2 * logsigma))    # 11
            kl_terms[t] = 0.5 * torch.sum(mu_2+sigma_2-2 * logsigma,1) - self.T * 0.5
            Lz += kl_terms[t]
        # Lz -= self.T / 2
        Lz = torch.mean(Lz)    ####################################################
        loss = Lz + Lx    # 12
        return loss,x_history_new,COMP_new,l_new

class DrawModelRNLN(DrawModel):
    def __init__(self,T,A,B,z_size,N,dec_size,enc_size):
        super(DrawModelRNLN,self).__init__(T,A,B,z_size,N,dec_size,enc_size)
        self.enc_ln = nn.LayerNorm(enc_size)
        self.dec_ln = nn.LayerNorm(dec_size)

    def forward(self,x,x_history_stack=None,COMP_stack=None):
        self.batch_size = x.size()[0]
        h_dec_prev = Variable(torch.zeros(self.batch_size,self.dec_size))
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))

        x_history_new,COMP_new,l_new = [],[],[]

        enc_state = Variable(torch.zeros(self.batch_size,self.enc_size))
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))
        for t in range(self.T):
            c_prev = Variable(torch.zeros(self.batch_size,self.A * self.B)) if t == 0 else self.cs[t-1]
            x_hat = x - self.sigmoid(c_prev)     # 3
            r_t = self.read(x,x_hat,h_dec_prev)
            h_enc_prev,enc_state = self.encoder(torch.cat((r_t,h_dec_prev),1),(h_enc_prev,enc_state))
            
            x_history_stats,COMP = getHistoryStats(h_enc_prev,x_history_stack,COMP_stack,t*2)       
            h_enc_prev, COMP, l = getMDL(h_enc_prev,x_history_stats,COMP,True)
            COMP_new.append(COMP)
            l_new.append(l)
            x_history_new.append(x_history_stats)

            h_enc_prev = self.enc_ln(h_enc_prev)

            # h_enc = self.encoder_gru(torch.cat((r_t,h_dec_prev),1),h_enc_prev)
            z,self.mus[t],self.logsigmas[t],self.sigmas[t] = self.sampleQ(h_enc_prev)
            h_dec,dec_state = self.decoder(z, (h_dec_prev, dec_state))

            x_history_stats,COMP = getHistoryStats(h_dec,x_history_stack,COMP_stack,t*2+1)       
            h_dec,COMP,l = getMDL(h_dec,x_history_stats,COMP,True)
            COMP_new.append(COMP)
            l_new.append(l)
            x_history_new.append(x_history_stats)

            h_dec = self.dec_ln(h_dec)

            # h_dec = self.decoder_gru(z, h_dec_prev)
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec
        
        return x_history_new,COMP_new,l_new

    def loss(self,x,x_history_stack=None,COMP_stack=None):
        x_history_new,COMP_new,l_new = self.forward(x,x_history_stack,COMP_stack)
        criterion = nn.BCELoss()
        x_recons = self.sigmoid(self.cs[-1])
        Lx = criterion(x_recons,x) * self.A * self.B
        Lz = 0
        kl_terms = [0] * T
        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]
            # Lz += (0.5 * (mu_2 + sigma_2 - 2 * logsigma))    # 11
            kl_terms[t] = 0.5 * torch.sum(mu_2+sigma_2-2 * logsigma,1) - self.T * 0.5
            Lz += kl_terms[t]
        # Lz -= self.T / 2
        Lz = torch.mean(Lz)    ####################################################
        loss = Lz + Lx    # 12
        return loss,x_history_new,COMP_new,l_new

class DrawModelSN(DrawModel):
    def __init__(self,T,A,B,z_size,N,dec_size,enc_size):
        super(DrawModelSN,self).__init__(T,A,B,z_size,N,dec_size,enc_size)
        # self.enc_ln = nn.LayerNorm(enc_size)
        # self.dec_ln = nn.LayerNorm(dec_size)

    def forward(self,x,y=None,x_history_stack=None,COMP_stack=None):
        self.batch_size = x.size()[0]
        h_dec_prev = Variable(torch.zeros(self.batch_size,self.dec_size))
        h_enc_prev = Variable(torch.zeros(self.batch_size, self.enc_size))

        x_history_new,COMP_new,l_new = [],[],[]

        s_enc = torch.ones((self.batch_size,self.enc_size))
        s_dec = torch.ones((self.batch_size,self.dec_size))
        if y is not None:
            for label in np.arange(10):
                s_enc[y==label] = len((y==label).nonzero()) / len(y)
                s_dec[y==label] = len((y==label).nonzero()) / len(y)

        enc_state = Variable(torch.zeros(self.batch_size,self.enc_size))
        dec_state = Variable(torch.zeros(self.batch_size, self.dec_size))
        for t in range(self.T):
            c_prev = Variable(torch.zeros(self.batch_size,self.A * self.B)) if t == 0 else self.cs[t-1]
            x_hat = x - self.sigmoid(c_prev)     # 3
            r_t = self.read(x,x_hat,h_dec_prev)
            h_enc_prev,enc_state = self.encoder(torch.cat((r_t,h_dec_prev),1),(h_enc_prev,enc_state))
            # h_enc_prev = self.enc_ln(h_enc_prev)
            
            x_history_stats,COMP = getHistoryStats(h_enc_prev,x_history_stack,COMP_stack,t*2)       
            h_enc_prev, COMP, l = getMDL(h_enc_prev,x_history_stats,COMP,RN=True,s=s_enc)
            COMP_new.append(COMP)
            l_new.append(l)
            x_history_new.append(x_history_stats)

            # h_enc = self.encoder_gru(torch.cat((r_t,h_dec_prev),1),h_enc_prev)
            z,self.mus[t],self.logsigmas[t],self.sigmas[t] = self.sampleQ(h_enc_prev)
            h_dec,dec_state = self.decoder(z, (h_dec_prev, dec_state))
            # h_dec = self.dec_ln(h_dec)

            x_history_stats,COMP = getHistoryStats(h_dec,x_history_stack,COMP_stack,t*2+1)       
            h_dec,COMP,l = getMDL(h_dec,x_history_stats,COMP,RN=True,s=s_dec)
            COMP_new.append(COMP)
            l_new.append(l)
            x_history_new.append(x_history_stats)

            # h_dec = self.decoder_gru(z, h_dec_prev)
            self.cs[t] = c_prev + self.write(h_dec)
            h_dec_prev = h_dec
        
        return x_history_new,COMP_new,l_new

    def loss(self,x,y=None,x_history_stack=None,COMP_stack=None):
        x_history_new,COMP_new,l_new = self.forward(x,y,x_history_stack,COMP_stack)
        criterion = nn.BCELoss()
        x_recons = self.sigmoid(self.cs[-1])
        Lx = criterion(x_recons,x) * self.A * self.B
        Lz = 0
        kl_terms = [0] * T
        for t in range(self.T):
            mu_2 = self.mus[t] * self.mus[t]
            sigma_2 = self.sigmas[t] * self.sigmas[t]
            logsigma = self.logsigmas[t]
            # Lz += (0.5 * (mu_2 + sigma_2 - 2 * logsigma))    # 11
            kl_terms[t] = 0.5 * torch.sum(mu_2+sigma_2-2 * logsigma,1) - self.T * 0.5
            Lz += kl_terms[t]
        # Lz -= self.T / 2
        Lz = torch.mean(Lz)    ####################################################
        loss = Lz + Lx    # 12
        return loss,x_history_new,COMP_new,l_new


# model = DrawModel(10,5,5,10,5,128,128)
# x = Variable(torch.ones(4,25))
# x_hat = Variable(torch.ones(4,25)*2)
# r = model.write()
# print r
# g = Variable(torch.ones(4,1))
# delta = Variable(torch.ones(4,1)  * 3)
# sigma = Variable(torch.ones(4,1))
# rng = Variable(torch.arange(0,5).view(1,-1))
# mu_x = model.compute_mu(g,rng,delta)
# a = Variable(torch.arange(0,5).view(1,1,-1))
# mu_x = mu_x.view(-1,5,1)
# sigma = sigma.view(-1,1,1)
# F = model.filterbank_matrices(a,mu_x,sigma)
# print F
# def test_normalSample():
#     print model.normalSample()
#
# def test_write():
#     h_dec = Variable(torch.zeros(8,128))
#     model.write(h_dec)
#
# def test_read():
#     x = Variable(torch.zeros(8,28*28))
#     x_hat = Variable((torch.zeros(8,28*28)))
#     h_dec = Variable(torch.zeros(8, 128))
#     model.read(x,x_hat,h_dec)
#
# def test_loss():
#     x = Variable(torch.zeros(8,28*28))
#     loss = model.loss(x)
#     print loss

