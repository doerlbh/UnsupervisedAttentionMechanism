import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

device = "cpu"
dtype = torch.FloatTensor


class QNetwork_UAM(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64, isTest=False):
        super(QNetwork_UAM, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
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
        out = x
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        mdl = [l_input, l_fc1, l_fc2]
        
        return out, x_history_stack, COMP_stack, mdl

class QNetworkLN_UAM(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64, isTest=False):
        super(QNetworkLN_UAM, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.isTest = isTest    

        self.input_ln = nn.LayerNorm(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc1_ln = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2_ln = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)    
        
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
        x = self.fc1_ln(x)
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
        x = self.fc2_ln(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = x
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        mdl = [l_input, l_fc1, l_fc2]
        
        return out, x_history_stack, COMP_stack, mdl

class QNetworkRN_UAM(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64, isTest=False):
        super(QNetworkRN_UAM, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
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
        out = x
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        mdl = [l_input, l_fc1, l_fc2]
        
        return out, x_history_stack, COMP_stack, mdl

class QNetworkRLN_UAM(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64, isTest=False):
        super(QNetworkRLN_UAM, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
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
        # l_nml_x = l_nml_x.expand(x.shape[0],x.shape[1])
        x = torch.mul(l_nml_x.expand(x.shape[0],x.shape[1]),x)
        return x, COMP, l_nml_x.mean()

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
            x, COMP_input, l_input = self.RegularityLN(x,x_history_stats,COMP)
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
        x, COMP_fc1, l_fc1 = self.RegularityLN(x,x_history_stats,COMP)
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
        x, COMP_fc2, l_fc2 = self.RegularityLN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = x
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        mdl = [l_input, l_fc1, l_fc2]
        
        return out, x_history_stack, COMP_stack, mdl

class QNetworkRNLN_UAM(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64, isTest=False):
        super(QNetworkRNLN_UAM, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.isTest = isTest    

        self.input_ln = nn.LayerNorm(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc1_ln = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2_ln = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)    
        
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
        x = self.fc1_ln(x)
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
        x = self.fc2_ln(x)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = x
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        mdl = [l_input, l_fc1, l_fc2]
        
        return out, x_history_stack, COMP_stack, mdl

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class QNetworkBN(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64):
        super(QNetworkBN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.input_bn = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc1_bn = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2_bn = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)    

    def forward(self, x):
        
        # x = self.input_bn(x)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x

class QNetworkLN(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64):
        super(QNetworkLN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.input_ln = nn.LayerNorm(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc1_ln = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc2_ln = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)    

    def forward(self, x):
        
        # x = self.input_ln(x)
        x = self.fc1(x)
        x = self.fc1_ln(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.fc2_ln(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x


class QNetworkWN(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64):
        super(QNetworkWN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = weightNorm(nn.Linear(state_size, fc1_units),name = "weight")
        self.fc2 = weightNorm(nn.Linear(fc1_units, fc2_units),name = "weight")
        self.fc3 = weightNorm(nn.Linear(fc2_units, action_size),name = "weight")    

    def forward(self, x):
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        return x


class QNetworkRN(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64, isTest=False):
        super(QNetworkRN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
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
            COMP = COMP_stack[2]
        x_history_fc2 = x_history_stats
        COMP_fc2 = -1
        x, COMP_fc2 = self.RegularityEN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = x
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        
        return out, x_history_stack, COMP_stack


class QNetworkRNLN(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64, isTest=False):
        super(QNetworkRNLN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.isTest = isTest    
        
        self.input_ln = nn.LayerNorm(state_size)
        self.fc1_ln = nn.LayerNorm(fc1_units)
        self.fc2_ln = nn.LayerNorm(fc2_units)   
        
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
        return x, COMP

    def forward(self, x, x_history_stack=None,COMP_stack=None,inputRN=False, isTest=False):
        
        self.isTest = isTest
        
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
            COMP = COMP_stack[2]
        x_history_fc2 = x_history_stats
        COMP_fc2 = -1
        x, COMP_fc2 = self.RegularityEN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = x
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        
        return out, x_history_stack, COMP_stack


class QNetworkRLN(nn.Module):
    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=64, isTest=False):
        super(QNetworkRLN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
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
        return x, COMP


    def forward(self, x, x_history_stack=None,COMP_stack=None,inputRN=False, isTest=False):
        
        self.isTest = isTest
        
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
            COMP = COMP_stack[2]
        x_history_fc2 = x_history_stats
        COMP_fc2 = -1
        x, COMP_fc2 = self.RegularityLN(x,x_history_stats,COMP)
        x = F.relu(x)
        
        x = self.fc3(x)
        out = x
        
        x_history_stack = [x_history_input, x_history_fc1, x_history_fc2]
        COMP_stack = [COMP_input, COMP_fc1, COMP_fc2]
        
        return out, x_history_stack, COMP_stack
