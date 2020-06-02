##########################
# Modified from https://github.com/chenzhaomin123/draw_pytorch

# by Baihan Lin for Unsupervised Attention Mechanism
# https://arxiv.org/abs/1902.10658

# Environment: Python 3.6+, Pytorch 1.0
##########################

import torch.optim as optim
from torchvision import datasets,transforms
import torch.utils
from draw_model import DrawModel,DrawModelLN,DrawModelRN,DrawModelRNLN,DrawModelSN
from config import *
from utility import Variable,save_image,xrecons_grid
import torch.nn.utils
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--alg', type=str, default="NN1", help='algorithm')
opt = parser.parse_args()

name = opt.alg

if "NN" in name: 
    model = DrawModel(T,A,B,z_size,N,dec_size,enc_size)
if "LN" in name and "RN" not in name:
    model = DrawModelLN(T,A,B,z_size,N,dec_size,enc_size)
if "RN" in name and "LN" not in name:
    model = DrawModelRN(T,A,B,z_size,N,dec_size,enc_size)
if "RN" in name and "LN" in name:
    model = DrawModelRNLN(T,A,B,z_size,N,dec_size,enc_size)
if "SN" in name:
    model = DrawModelSN(T,A,B,z_size,N,dec_size,enc_size)

optimizer = optim.Adam(model.parameters(),lr=learning_rate,betas=(beta1,0.999))
# optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=0)

if USE_CUDA: model.cuda()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data/', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor()])),
    batch_size=batch_size, shuffle=False)

def train():
    f = open("score/"+name+"_loss.csv", "w")
    avg_loss = 0
    count = 0
    x_history_stack,COMP_stack = None,None
    for epoch in range(epoch_num):
        for data,target in train_loader:
            bs = data.size()[0]
            data = Variable(data).view(bs, -1).to(device)
            optimizer.zero_grad()
            if "RN" in name:
                loss,x_history_stack,COMP_stack,MDL = model.loss(data,x_history_stack,COMP_stack)
            elif "SN" in name:
                loss,x_history_stack,COMP_stack,MDL = model.loss(data,target,x_history_stack,COMP_stack)            
            else:
                loss = model.loss(data)
            avg_loss += loss.cpu().data.numpy()
            loss.backward(retain_graph=True)
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
            count += 1
            if count % 10 == 0:
                print('Epoch-{}; Count-{}; loss: {};'.format(epoch, count, avg_loss / 10))
                f.write(str(avg_loss/10)+"\n")
                if count % 500 == 0:
                    torch.save(model.state_dict(),'save/'+name+'weights_%d.tar'%(count))
                    # generate_image(count)
                avg_loss = 0
    torch.save(model.state_dict(), 'save/'+name+'weights_final.tar')
    f.close()
    # generate_image(count)


def generate_image(count):
    x = model.generate(batch_size)
    save_image(x,count)

def save_example_image():
    train_iter = iter(train_loader)
    data, _ = next(train_iter)
    img = data.cpu().numpy().reshape(batch_size, 28, 28)
    imgs = xrecons_grid(img, B, A)
    plt.matshow(imgs, cmap=plt.cm.gray)
    plt.savefig('image/example.png')

if __name__ == '__main__':
    # save_example_image()
    train()