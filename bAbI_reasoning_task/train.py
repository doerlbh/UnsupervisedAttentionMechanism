##########################
# Modified from https://github.com/chingyaoc/ggnn.pytorch

# by Baihan Lin for Unsupervised Attention Mechanism
# https://arxiv.org/abs/1902.10658

# Environment: Python 3.6, Pytorch 1.0
##########################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import data
import model
import config
import utils
import structure

# train
def train(opt, test=False, trained_model=None,hist=None):
    output = structure.Output()
    mode = 'test' if test else 'train'
    babisplit, babiloader = data.get_loader(mode=mode, task_id=opt.task_id, data_id=opt.data_id, train_size=opt.train_size, val=False, ques_id=opt.question_id)
    opt.n_edge_type = babisplit.n_edge_type
    opt.n_node_type = babisplit.n_node_type
    opt.n_label_type = babisplit.n_label_type
    
    if hist is not None:
        xhistory,comp = hist
    else:
        xhistory,comp = None,None
    
    isTrain = not test

    if test or trained_model is not None:
        net = trained_model
    else:
        if opt.alg == "ggnn":
            net = model.GGNN(opt).to(config.device)
        elif opt.alg == "ggnnrn":
            net = model.GGNNRN(opt).to(config.device)
        elif opt.alg == "ggnnln":
            net = model.GGNNLN(opt).to(config.device)
        elif opt.alg == "ggnnrnln":
            net = model.GGNNRNLN(opt).to(config.device)
    # net = net.double()
    if opt.resume:
        logs = torch.load(config.MODEL_PATH)
        net.load_state_dict(logs['weights'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    # optimizer = optim.SGD(net.parameters(), lr=config.LEARNING_RATE)

    if test:
        net.eval()
    else:
        net.train()
        output.set_net(net)

    # if config.VERBAL and not test:
        # print('------------------------ Dataset: '+str(opt.data_id)+' -------------------------------')

    num_epoch = 1 if test else opt.epoch
    for i in range(num_epoch):
        total_loss = []
        total_accuracy = []
        for adj_matrix, annotation, target in babiloader:
            padding = torch.zeros(len(annotation), opt.n_node_type, opt.hidden_dim - config.ANNOTATION_DIM[str(opt.task_id)])
            # padding = padding.double()
            annotation = annotation.float()
            x = torch.cat((annotation, padding), 2)

            x = Variable(x.to(config.device))
            m = Variable(adj_matrix.to(config.device))
            a = Variable(annotation.to(config.device))
            t = Variable(target.to(config.device)).long()

            if "rn" in opt.alg:
                pred,xhistory,comp,mdl = net(x, a, m,xhistory,comp,isTrain)
            else:
                pred = net(x, a, m)
            if opt.task_id == 19:
                # consider each step as a prediction
                pred = pred.view(-1, pred.shape[-1])
                t = t.view(-1)
            loss = criterion(pred, t)
            if not test:
                net.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
            total_loss.append(loss.item())

            accuracy = (pred.max(1)[1] == t).float().mean()
            total_accuracy.append(accuracy.item())

        if config.VERBAL:
            print(mode + ' Epoch: ' + str(i) + ' Loss: {:.3f} '.format(sum(total_loss) / len(total_loss)) + ' Accuracy: {:.3f} '.format(sum(total_accuracy) / len(total_accuracy)))
        output.set_loss(sum(total_loss) / len(total_loss))
        output.set_accuracy(sum(total_accuracy) / len(total_accuracy))

    return output