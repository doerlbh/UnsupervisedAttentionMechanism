##########################
# Modified from https://github.com/chingyaoc/ggnn.pytorch

# by Baihan Lin for Unsupervised Attention Mechanism
# https://arxiv.org/abs/1902.10658

# Environment: Python 3.6, Pytorch 1.0
##########################

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import config
import utils
import model
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--train_size', type=int, default=100, help='number of instance we used to train, 1~1000')
parser.add_argument('--task_id', type=int, default=4, help='bAbI task id')
parser.add_argument('--question_id', type=int, default=0, help='question id for those tasks have several types of questions')
parser.add_argument('--data_id', type=int, default=0, help='generated bAbI data id 1~10')
parser.add_argument('--hidden_dim', type=int, default=10, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')
parser.add_argument('--epoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--outer_epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--resume', action='store_true', help='resume a pretrained model')
parser.add_argument('--name', type=str, default='model', help='name of model')
parser.add_argument('--alg', type=str, default='ggnn', help='name of alg')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')

opt = parser.parse_args()
if config.VERBAL: print(opt)

def main(opt):
    average_accuracy = 0

    if opt.data_id == 0:
        f = open("score/"+opt.name+"_acc.csv", "w")
        # rum experiment 10 times using 10 different generated dataset
        n_syn = 10
        net,xhist,comp = [None]*n_syn,[None]*n_syn,[None]*n_syn
        for j in range(opt.outer_epoch):
            acc = []
            average_accuracy = 0
            for i in range(n_syn):
                opt.data_id = i + 1
                train_output = train(opt, trained_model=net[i],hist=[xhist[i],comp[i]])
                test_output =  train(opt, test=True, trained_model=train_output.get_net())
                net[i] = train_output.get_net()
                xhist[i],comp[i] = train_output.get_history()
                acc.append(test_output.get_accuracy())
                average_accuracy += test_output.get_accuracy()
            average_accuracy = average_accuracy / n_syn
            score_mean = np.mean(acc)
            score_std = np.std(acc)
            print("EPOCH: ",j,"====================",acc)
            f.write(str(score_mean)+","+str(score_std)+"\n")
        f.close()
    else:
        # run experiment one time at data_id
        net,xhist,comp = None,None,None
        for j in range(opt.outer_epoch):
            train_output = train(opt, trained_model=net,hist=[xhist,comp])
            test_output =  train(opt, test=True, trained_model=train_output.get_net())
            net = train_output.get_net()
            xhist,comp = train_output.get_history()
        average_accuracy += test_output.get_accuracy()
    
    print('Test accuracy is: ' + str(average_accuracy) + ' for task: ' + str(opt.task_id) + ' at question: ' + str(opt.question_id) + ' using Num: ' + str(opt.train_size) + ' training data.' )

    results = {
        'train_size': opt.train_size,
        'task_id': opt.task_id,
        'question_id': opt.question_id,
        'data_id': opt.data_id,
        'hidden_dim': opt.hidden_dim,
        'n_steps': opt.n_steps,
        'epoch': opt.epoch,
        'weights': train_output.get_net().state_dict(),
    }
    torch.save(results, os.path.join('model', '{}.pth'.format(opt.name)))

    return None

if __name__ == "__main__":
    main(opt)