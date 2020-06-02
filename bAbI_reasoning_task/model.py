##########################
# Modified from https://github.com/chingyaoc/ggnn.pytorch

# by Baihan Lin for Unsupervised Attention Mechanism
# https://arxiv.org/abs/1902.10658

# Environment: Python 3.6, Pytorch 1.0
##########################

import config
import torch
import torch.nn as nn

import time

# if config.device == "cuda":
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')

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
    log_p_x_thetax = p_thetax.log_prob(x).to(config.device)        
    log_p_x_max = torch.max(log_p_x_thetax).float().to(config.device)
    sum_p_x_thetax = log_p_x_max + torch.log(torch.sum(torch.exp(log_p_x_thetax - log_p_x_max))).to(config.device)
    COMP_max = torch.max(COMP).to(config.device)
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
        COMP = torch.tensor(0.0).to(config.device)
    else:
        if isTest:
            x_history_stats = x_history_stack[l] 
        else:
            x_history_stats = [(x_history_stack[l][0]*x_history_stack[l][2]+x_stats[0]*x_stats[2])/(x_history_stack[l][2]+x_stats[2]),torch.sqrt(((x_history_stack[l][2]-1)*torch.pow(x_history_stack[l][1],2)+(x_stats[2]-1)*torch.pow(x_stats[1],2))/(x_history_stack[l][2]+x_stats[2]-2)),x_history_stack[l][2]+x_stats[2]]
        COMP = COMP_stack[l]
    return x_history_stats, COMP

class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(GGNN, self).__init__()
        self.task_id = opt.task_id
        self.hidden_dim = opt.hidden_dim
        self.annotation_dim = config.ANNOTATION_DIM[str(opt.task_id)]
        self.n_node = opt.n_node_type
        self.n_edge = opt.n_edge_type
        self.n_output = opt.n_label_type
        self.n_steps = opt.n_steps

        self.fc_in = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_edge)
        self.fc_out = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_edge)

        self.gated_update = GatedPropagation(self.hidden_dim, self.n_node, self.n_edge)

        if self.task_id == 18 or self.task_id == 19:
            self.graph_aggregate =  GraphFeature(self.hidden_dim, self.n_node, self.n_edge, self.annotation_dim)
            self.fc_output = nn.Linear(self.hidden_dim, self.n_output)
        else:
            self.fc1 = nn.Linear(self.hidden_dim+self.annotation_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, 1)
            self.tanh = nn.Tanh()

    def forward(self, x, a, m):
        '''
        init state x: [batch_size, num_node, hidden_size] , pad zero from annoatation
        annoatation x: [batch_size, num_node, 1] 
        adj matrix m: [batch_size, num_node, num_node * n_edge_types * 2]
        output out: [batch_size, n_label_types], for task 4, 15, 16, n_label_types == num_nodes
        '''
        # x, a, m = x.double(), a.double(), m.double()
        all_x = [] # used for task 19, to track 
        for i in range(self.n_steps):
            in_states = self.fc_in(x)
            out_states = self.fc_out(x)
            in_states = in_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            out_states = out_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            x = self.gated_update(in_states, out_states, x, m)
            all_x.append(x)

        if self.task_id == 18:
            output = self.graph_aggregate(torch.cat((x, a), 2))
            output = self.fc_output(output)
        elif self.task_id == 19:
            step1 = self.graph_aggregate(torch.cat((all_x[0], a), 2))
            step1 = self.fc_output(step1).view(-1,1,self.n_output)
            step2 = self.graph_aggregate(torch.cat((all_x[1], a), 2))
            step2 = self.fc_output(step2).view(-1,1,self.n_output)
            output = torch.cat((step1,step2), 1)
        else:
            output = self.fc1(torch.cat((x, a), 2))
            output = self.tanh(output)
            output = self.fc2(output).sum(2)
        return output


class GraphFeature(nn.Module):
    '''
    Output a Graph-Level Feature
    '''
    def __init__(self, hidden_dim, n_node, n_edge, n_anno):
        super(GraphFeature, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_node = n_node
        self.n_edge = n_edge
        self.n_anno = n_anno

        self.fc_i = nn.Linear(self.hidden_dim + self.n_anno, self.hidden_dim)
        self.fc_j = nn.Linear(self.hidden_dim + self.n_anno, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''
        input x: [batch_size, num_node, hidden_size + annotation]
        output x: [batch_size, hidden_size]
        '''
        x_sigm = self.sigmoid(self.fc_i(x))
        x_tanh = self.tanh(self.fc_j(x))
        x_new = (x_sigm * x_tanh).sum(1)

        return self.tanh(x_new)


class GatedPropagation(nn.Module):
    '''
    Gated Recurrent Propagation
    '''
    def __init__(self, hidden_dim, n_node, n_edge):
        super(GatedPropagation, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_node = n_node
        self.n_edge = n_edge

        self.gate_r = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.gate_z = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.trans  = nn.Linear(self.hidden_dim*3, self.hidden_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_in, x_out, x_curt, matrix):
        matrix_in  = matrix[:, :, :self.n_node*self.n_edge].float()
        matrix_out = matrix[:, :, self.n_node*self.n_edge:].float()

        a_in  = torch.bmm(matrix_in, x_in)
        a_out = torch.bmm(matrix_out, x_out)
        a = torch.cat((a_in, a_out, x_curt), 2)

        z = self.sigmoid(self.gate_z(a))
        r = self.sigmoid(self.gate_r(a))

        joint_input = torch.cat((a_in, a_out, r * x_curt), 2)
        h_hat = self.tanh(self.trans(joint_input))
        output = (1 - z) * x_curt + z * h_hat

        return output


class GGNNLN(GGNN):
    """
    Gated Graph Sequence Neural Networks (GGNN) w/ LN
    """
    def __init__(self, opt):
        super(GGNNLN, self).__init__(opt)

        self.fc_in_ln = nn.LayerNorm(self.hidden_dim * self.n_edge)
        self.fc_out_ln = nn.LayerNorm(self.hidden_dim * self.n_edge)

    def forward(self, x, a, m):
        '''
        init state x: [batch_size, num_node, hidden_size] , pad zero from annoatation
        annoatation x: [batch_size, num_node, 1] 
        adj matrix m: [batch_size, num_node, num_node * n_edge_types * 2]
        output out: [batch_size, n_label_types], for task 4, 15, 16, n_label_types == num_nodes
        '''
        # x, a, m = x.double(), a.double(), m.double()
        all_x = [] # used for task 19, to track 
        for i in range(self.n_steps):
            in_states = self.fc_in(x)
            in_states = self.fc_in_ln(in_states)
            out_states = self.fc_out(x)
            out_states = self.fc_out_ln(out_states)
            in_states = in_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            out_states = out_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            x = self.gated_update(in_states, out_states, x, m)
            all_x.append(x)

        if self.task_id == 18:
            output = self.graph_aggregate(torch.cat((x, a), 2))
            output = self.fc_output(output)
        elif self.task_id == 19:
            step1 = self.graph_aggregate(torch.cat((all_x[0], a), 2))
            step1 = self.fc_output(step1).view(-1,1,self.n_output)
            step2 = self.graph_aggregate(torch.cat((all_x[1], a), 2))
            step2 = self.fc_output(step2).view(-1,1,self.n_output)
            output = torch.cat((step1,step2), 1)
        else:
            output = self.fc1(torch.cat((x, a), 2))
            output = self.tanh(output)
            output = self.fc2(output).sum(2)
        return output

class GGNNRN(GGNN):
    """
    Gated Graph Sequence Neural Networks (GGNN) w/ RN
    """
    def __init__(self, opt):
        super(GGNNRN, self).__init__(opt)
        
        self.fc_in_ln = nn.LayerNorm(self.hidden_dim * self.n_edge)
        self.fc_out_ln = nn.LayerNorm(self.hidden_dim * self.n_edge)

    def forward(self, x, a, m, x_history_stack=None,COMP_stack=None,isTrain=True):
        '''
        init state x: [batch_size, num_node, hidden_size] , pad zero from annoatation
        annoatation x: [batch_size, num_node, 1] 
        adj matrix m: [batch_size, num_node, num_node * n_edge_types * 2]
        output out: [batch_size, n_label_types], for task 4, 15, 16, n_label_types == num_nodes
        '''
        # x, a, m = x.double(), a.double(), m.double()
        all_x = [] # used for task 19, to track 
         
        # if x_history_stack is not None:
        #     x_history_stack,COMP_stack = x_history_stack.to(config.device),COMP_stack.to(config.device)
        x_history_new,COMP_new,l_new = [],[],[]

        for i in range(self.n_steps):
            in_states = self.fc_in(x)

            x_history_stats,COMP = getHistoryStats(in_states,x_history_stack,COMP_stack,i*2)       
            in_states, COMP, l = getMDL(in_states,x_history_stats,COMP,isTrain)
            COMP_new.append(COMP)
            l_new.append(l)
            x_history_new.append(x_history_stats)

            # in_states = self.fc_in_ln(in_states)

            out_states = self.fc_out(x)

            x_history_stats,COMP = getHistoryStats(out_states,x_history_stack,COMP_stack,i*2+1)       
            out_states,COMP,l = getMDL(out_states,x_history_stats,COMP,isTrain)
            COMP_new.append(COMP)
            l_new.append(l)
            x_history_new.append(x_history_stats)

            # out_states = self.fc_out_ln(out_states)

            in_states = in_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            out_states = out_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge, self.hidden_dim) 
            x = self.gated_update(in_states, out_states, x, m)
            all_x.append(x)

        if self.task_id == 18:
            output = self.graph_aggregate(torch.cat((x, a), 2))
            output = self.fc_output(output)
        elif self.task_id == 19:
            step1 = self.graph_aggregate(torch.cat((all_x[0], a), 2))
            step1 = self.fc_output(step1).view(-1,1,self.n_output)
            step2 = self.graph_aggregate(torch.cat((all_x[1], a), 2))
            step2 = self.fc_output(step2).view(-1,1,self.n_output)
            output = torch.cat((step1,step2), 1)
        else:
            output = self.fc1(torch.cat((x, a), 2))
            output = self.tanh(output)
            output = self.fc2(output).sum(2)
        return output, x_history_new, COMP_new, l_new

class GGNNRNLN(GGNN):
    """
    Gated Graph Sequence Neural Networks (GGNN) w/ RN
    """
    def __init__(self, opt):
        super(GGNNRNLN, self).__init__(opt)
        
        self.fc_in_ln = nn.LayerNorm(self.hidden_dim * self.n_edge)
        self.fc_out_ln = nn.LayerNorm(self.hidden_dim * self.n_edge)

    def forward(self, x, a, m, x_history_stack=None,COMP_stack=None,isTrain=True):
        '''
        init state x: [batch_size, num_node, hidden_size] , pad zero from annoatation
        annoatation x: [batch_size, num_node, 1] 
        adj matrix m: [batch_size, num_node, num_node * n_edge_types * 2]
        output out: [batch_size, n_label_types], for task 4, 15, 16, n_label_types == num_nodes
        '''
        # x, a, m = x.double(), a.double(), m.double()
        all_x = [] # used for task 19, to track 

        x_history_new,COMP_new,l_new = [],[],[]

        for i in range(self.n_steps):
            in_states = self.fc_in(x)

            x_history_stats,COMP = getHistoryStats(in_states,x_history_stack,COMP_stack,i*2)       
            in_states, COMP, l = getMDL(in_states,x_history_stats,COMP,isTrain)
            COMP_new.append(COMP)
            l_new.append(l)
            x_history_new.append(x_history_stats)

            in_states = self.fc_in_ln(in_states)

            out_states = self.fc_out(x)

            x_history_stats,COMP = getHistoryStats(out_states,x_history_stack,COMP_stack,i*2+1)       
            out_states,COMP,l = getMDL(out_states,x_history_stats,COMP,isTrain)
            COMP_new.append(COMP)
            l_new.append(l)
            x_history_new.append(x_history_stats)

            out_states = self.fc_out_ln(out_states)

            in_states = in_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            out_states = out_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge, self.hidden_dim) 
            x = self.gated_update(in_states, out_states, x, m)
            all_x.append(x)

        if self.task_id == 18:
            output = self.graph_aggregate(torch.cat((x, a), 2))
            output = self.fc_output(output)
        elif self.task_id == 19:
            step1 = self.graph_aggregate(torch.cat((all_x[0], a), 2))
            step1 = self.fc_output(step1).view(-1,1,self.n_output)
            step2 = self.graph_aggregate(torch.cat((all_x[1], a), 2))
            step2 = self.fc_output(step2).view(-1,1,self.n_output)
            output = torch.cat((step1,step2), 1)
        else:
            output = self.fc1(torch.cat((x, a), 2))
            output = self.tanh(output)
            output = self.fc2(output).sum(2)
        return output, x_history_new, COMP_new, l_new

