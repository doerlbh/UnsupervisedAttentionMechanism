##########################
# Modified from https://github.com/chingyaoc/ggnn.pytorch

# by Baihan Lin for Unsupervised Attention Mechanism
# https://arxiv.org/abs/1902.10658

# Environment: Python 3.6, Pytorch 1.0
##########################

# Define some useful structure
class Output(object):
    def __init__(self):
        self.accuracy = 0
        self.loss = 0
        self.net = None
        self.xhist = None
        self.comp = None
    
    def set_accuracy(self, value):
        self.accuracy = value

    def set_loss(self, value):
        self.loss = value

    def set_net(self, net):
        self.net = net

    def get_accuracy(self):
        return self.accuracy

    def get_loss(self):
        return self.loss

    def get_net(self):
        return self.net
    
    def se_history(self,xh,cp):
        self.xhist, self.comp = xh, cp

    def get_history(self):
        return self.xhist, self.comp