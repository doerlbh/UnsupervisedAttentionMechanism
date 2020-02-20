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

DOWNLOAD_MNIST = True   # set to True if haven't download the data


def prepare_data_mnist():
    transformer = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.MNIST(root='./mnist',download=DOWNLOAD_MNIST,train=True, transform=transforms.Compose([
                           transforms.ToTensor()
                           ,transforms.Normalize((0.1307,), (0.3081,))
                       ]))
    test_data = datasets.MNIST(root='./mnist',download=DOWNLOAD_MNIST,train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                           ,transforms.Normalize((0.1307,), (0.3081,))
                       ]))
#     train_data = datasets.MNIST(root='./mnist',download=DOWNLOAD_MNIST,train=True, transform=transformer)
#     test_data = datasets.MNIST(root='./mnist',download=DOWNLOAD_MNIST,train=False, transform=transformer)
    
    image_width = 28
    image_height = 28

    return train_data,test_data,image_width,image_height


train_data, test_data, IMAGE_WIDTH, IMAGE_HEIGHT = prepare_data_mnist()

num_train = len(train_data)
indices = list(range(num_train))

split = 5000
train_size = 60000 - split
valid_size = split
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=False)
valid_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=split, sampler=valid_sampler, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=False)

num_classes = 10
classe_labels = range(num_classes)


def prepare_data_cifar100():
    
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.507075159237, 0.486548873315, 0.440917843367), (1, 1, 1))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
    ])

    image_width = 32
    image_height = 32
    
    transformer = transforms.Compose([transforms.ToTensor()])
    train_data = datasets.CIFAR100(root='./cifar',download=False,train=True, transform=transform_train)
    test_data = datasets.CIFAR100(root='./cifar',download=False,train=False, transform=transform_test)
    return train_data,test_data,image_width,image_height


def vis_confusion(confusion_mtx, labels, figsize=(10, 10)):
    
    cm = confusion_mtx
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%' % p
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    
    fig = plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', cmap="Blues")
    plt.show()
    
    return fig


def train(seed,n_down,p_down,model, device, train_loader, train_size, valid_loader, valid_size, optimizer, epoch, train_losses, val_losses,isImbalanced=False,getMDL=False):
    tic = time.time()
    model.train()
    MDL = []
    
    if isImbalanced:
        torch.manual_seed(seed)  
        num_classes = 10
        classe_labels = range(num_classes)
        # sample_probs = torch.rand(num_classes)
        sample_probs = torch.ones(num_classes)   
        sample_probs[:n_down] = p_down
        sample_probs = sample_probs[torch.randperm(num_classes)]
        print('============================', sample_probs)
        idx_to_del = [i for i, label in enumerate(train_loader.dataset.train_labels) 
                      if random.random() > sample_probs[label]]
        imbalanced_train_dataset = copy.deepcopy(train_data)
        imbalanced_train_dataset.targets = np.delete(train_loader.dataset.train_labels.numpy(), np.array(idx_to_del), axis=0)
        imbalanced_train_dataset.data = np.delete(train_loader.dataset.train_data, np.array(idx_to_del), axis=0)
        imbalanced_train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_loader = imbalanced_train_loader
    
    used_train_size = len(train_loader.dataset)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if getMDL:
            output, _, _, l = model(data)
            MDL.append(l)
        else:
            output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 1 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
            train_loss = loss.item()
            it = batch_idx*len(data)
            percentage = 100.*it/used_train_size
        
            for batch_idx, (data, target) in enumerate(valid_loader):
                data, target = data.to(device), target.to(device)
                if getMDL:
                    output, _, _, _ = model(data)
                else:
                    output = model(data)
                loss = F.nll_loss(output, target)
                val_loss = loss.item()      
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                val_acc_100 = correct / valid_size
                
            print('epoch ', epoch, ': ', it, '/',used_train_size,' (%.0f' % percentage,'%)',' | train loss:%.4f' % train_loss, '| val loss:%.4f' % val_loss, '| val acc:%.4f' % val_acc_100, '| Time:%.4f' % (time.time()-tic))
            train_losses.append(train_loss)
            val_losses.append(val_loss)
    if getMDL:
        return train_losses, val_losses, used_train_size, MDL
    else:
        return train_losses, val_losses, used_train_size
            

def test(model, device, test_loader,test_acces,getMDL=False):
    model.eval()
    test_loss = 0
    correct = 0
    MDL = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if getMDL:
                output, _, _, l = model(data)
            else:
                output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * test_acc))
    test_acces.append(test_acc)
    
    confusion_mtx = sm.confusion_matrix(target.cpu().numpy(), pred.cpu().numpy())
    
    if getMDL:
        return test_acces, confusion_mtx, MDL
    else:
        return test_acces, confusion_mtx


def train_RN(seed,n_down,p_down,model, device, train_loader, train_size, valid_loader, valid_size, optimizer, epoch, train_losses, val_losses, x_history_stack=None,COMP_stack=None,isImbalanced=False,getMDL=False):
    tic = time.time()
    model.train()
    MDL = []

    if isImbalanced:
        torch.manual_seed(seed)
        num_classes = 10
        classe_labels = range(num_classes)
        # sample_probs = torch.rand(num_classes)
        sample_probs = torch.ones(num_classes)   
        sample_probs[:n_down] = p_down
        sample_probs = sample_probs[torch.randperm(num_classes)]
        print('============================', sample_probs)
        idx_to_del = [i for i, label in enumerate(train_loader.dataset.train_labels) 
                      if random.random() > sample_probs[label]]
        imbalanced_train_dataset = copy.deepcopy(train_data)
        imbalanced_train_dataset.targets = np.delete(train_loader.dataset.train_labels.numpy(), np.array(idx_to_del), axis=0)
        imbalanced_train_dataset.data = np.delete(train_loader.dataset.train_data, np.array(idx_to_del), axis=0)
        imbalanced_train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_loader = imbalanced_train_loader

    used_train_size = len(train_loader.dataset)
    
    # batch_idx = -1     
    for batch_idx, (data, target) in enumerate(train_loader):
        # batch_idx += 1
        data, target = data.to(device), target.to(device)
        # if x_history_stack is not None:
            # x_history_stack, COMP_stack = x_history_stack.to(device), COMP_stack.to(device)
        optimizer.zero_grad()
        if getMDL:
            output, x_history_stack, COMP_stack, l = model(data, x_history_stack, COMP_stack)
            MDL.append(l)
        else:
            output, x_history_stack, COMP_stack = model(data, x_history_stack, COMP_stack)
        loss = F.nll_loss(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        if batch_idx % 1 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
            train_loss = loss.item()
            it = batch_idx*len(data)
            percentage = 100.*it/used_train_size
        
            for batch_idx, (data, target) in enumerate(valid_loader):
                data, target = data.to(device), target.to(device)
                if getMDL:
                    output, x_history_stack_val, COMP_stack_val, _ = model(data, x_history_stack, COMP_stack,isTest=True)
                else:
                    output, x_history_stack_val, COMP_stack_val = model(data, x_history_stack, COMP_stack,isTest=True)
                loss = F.nll_loss(output, target)
                val_loss = loss.item()      
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                val_acc_100 = correct / valid_size
                
            print('epoch ', epoch, ': ', it, '/',used_train_size,' (%.0f' % percentage,'%)',' | train loss:%.4f' % train_loss, '| val loss:%.4f' % val_loss, '| val acc:%.4f' % val_acc_100, '| Time:%.4f' % (time.time()-tic))
            train_losses.append(train_loss)
            val_losses.append(val_loss)
    torch.cuda.empty_cache()
    if getMDL:
        return train_losses, val_losses, x_history_stack, COMP_stack, used_train_size, MDL
    else:
        return train_losses, val_losses, x_history_stack, COMP_stack, used_train_size
            
def test_RN(model, device, test_loader,test_acces,x_history_stack,COMP_stack,getMDL=False):
    model.eval()
    test_loss = 0
    correct = 0
    MDL = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # x_history_stack, COMP_stack = x_history_stack.to(device), COMP_stack.to(device)
            if getMDL:
                output, x_history_stack_test, COMP_stack_test, l = model(data, x_history_stack, COMP_stack,isTest=True)
                MDL.append(l)
            else:
                output, x_history_stack_test, COMP_stack_test = model(data, x_history_stack, COMP_stack,isTest=True)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * test_acc))
    test_acces.append(test_acc)
    
    confusion_mtx = sm.confusion_matrix(target.cpu().numpy(), pred.cpu().numpy())
    
    if getMDL:
        return test_acces, x_history_stack_test, COMP_stack_test, confusion_mtx, MDL
    else:
        return test_acces, x_history_stack_test, COMP_stack_test, confusion_mtx


def train_SN(seed,n_down,p_down,model, device, train_loader, train_size, valid_loader, valid_size, optimizer, epoch, train_losses, val_losses, x_history_stack=None,COMP_stack=None,isImbalanced=False,getMDL=False):
    tic = time.time()
    model.train()
    MDL = []

    if isImbalanced:
        torch.manual_seed(seed)
        num_classes = 10
        classe_labels = range(num_classes)
        # sample_probs = torch.rand(num_classes)
        sample_probs = torch.ones(num_classes)   
        sample_probs[:n_down] = p_down
        sample_probs = sample_probs[torch.randperm(num_classes)]
        print('============================', sample_probs)
        idx_to_del = [i for i, label in enumerate(train_loader.dataset.train_labels) 
                      if random.random() > sample_probs[label]]
        imbalanced_train_dataset = copy.deepcopy(train_data)
        imbalanced_train_dataset.targets = np.delete(train_loader.dataset.train_labels.numpy(), np.array(idx_to_del), axis=0)
        imbalanced_train_dataset.data = np.delete(train_loader.dataset.train_data, np.array(idx_to_del), axis=0)
        imbalanced_train_loader = torch.utils.data.DataLoader(imbalanced_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        train_loader = imbalanced_train_loader

    used_train_size = len(train_loader.dataset)
        
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)        
        # if x_history_stack is not None:
        #    x_history_stack, COMP_stack = x_history_stack.to(device), COMP_stack.to(device)
        optimizer.zero_grad()
        if getMDL:
            output, x_history_stack, COMP_stack, l = model(data, target, x_history_stack, COMP_stack)
            MDL.append(l)
        else:
            output, x_history_stack, COMP_stack = model(data, target, x_history_stack, COMP_stack)
        loss = F.nll_loss(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        if batch_idx % 1 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))
            train_loss = loss.item()
            it = batch_idx*len(data)
            percentage = 100.*it/used_train_size
        
            for batch_idx, (data, target) in enumerate(valid_loader):
                data, target = data.to(device), target.to(device)
                if getMDL:
                    output, x_history_stack_val, COMP_stack_val, _ = model(data, None, x_history_stack, COMP_stack,isTest=True)
                else:
                    output, x_history_stack_val, COMP_stack_val = model(data, None, x_history_stack, COMP_stack,isTest=True)
                loss = F.nll_loss(output, target)
                val_loss = loss.item()      
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(target.view_as(pred)).sum().item()
                val_acc_100 = correct / valid_size
                
            print('epoch ', epoch, ': ', it, '/',used_train_size,' (%.0f' % percentage,'%)',' | train loss:%.4f' % train_loss, '| val loss:%.4f' % val_loss, '| val acc:%.4f' % val_acc_100, '| Time:%.4f' % (time.time()-tic))
            train_losses.append(train_loss)
            val_losses.append(val_loss)
    torch.cuda.empty_cache()
    if getMDL:
        return train_losses, val_losses, x_history_stack, COMP_stack, used_train_size, MDL
    else:
        return train_losses, val_losses, x_history_stack, COMP_stack, used_train_size
            
def test_SN(model, device, test_loader,test_acces,x_history_stack,COMP_stack,getMDL=False):
    model.eval()
    test_loss = 0
    correct = 0
    MDL = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            # x_history_stack, COMP_stack = x_history_stack.to(device), COMP_stack.to(device)
            if getMDL:
                output, x_history_stack_test, COMP_stack_test, l = model(data, None, x_history_stack, COMP_stack,isTest=True)
                MDL.append(l)
            else:
                output, x_history_stack_test, COMP_stack_test = model(data, None, x_history_stack, COMP_stack,isTest=True)

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * test_acc))
    test_acces.append(test_acc)
    
    confusion_mtx = sm.confusion_matrix(target.cpu().numpy(), pred.cpu().numpy())
    if getMDL:
        return test_acces, x_history_stack_test, COMP_stack_test, confusion_mtx, MDL
    else:
        return test_acces, x_history_stack_test, COMP_stack_test, confusion_mtx


