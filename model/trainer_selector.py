"""
A rnn model for relation extraction, written in pytorch.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F

from utils import torch_utils
from model import layers

def idx_to_onehot(target, opt, val=0):
    sample_size, class_size = target.size(0), opt['num_class']

    y = torch.zeros(sample_size, class_size).fill_(val)
    y = y.scatter_(1, torch.unsqueeze(target.data, dim=1), 1.0 - val)
    y = Variable(y)

    return y

def sigmoid_loss(logits, target):
    batch_size = logits.size(0)
    logits = F.sigmoid(logits)
    ele_loss = target * torch.log(logits + 1e-20) + (1 - target) * torch.log(1 - logits + 1e-20)
    loss = -torch.sum(ele_loss) / batch_size
    return loss

# the relation model
class SelectorTrainer(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, selector):
        # options
        self.opt = opt
        # encoding model
        self.model = selector
        # all parameters of the model
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        # whether moving all data to gpu
        if opt['cuda']:
            self.model.cuda()
        # intialize the optimizer
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
    
    # train the model with a batch
    def update(self, inputs, target):
        """ Run a step of forward and backward model update. """
        # whether moving all data to gpu
        # inputs: [feature size, batch size, seq length]
        # labels: [batch size, seq length]
        #target = idx_to_onehot(target, self.opt)
        target = idx_to_onehot(target, self.opt, 0)
        if self.opt['cuda']:
            inputs = dict([(k, v.cuda()) for k, v in inputs.items()])
            target = target.cuda()

        # step forward
        # update the mode of the model
        self.model.train()
        # flush the optimizer
        self.optimizer.zero_grad()
        # forward calculation
        logits, _ = self.model(inputs)
        # calculate the loss
        loss = sigmoid_loss(logits, target)
        
        # backward
        # backward calculation
        loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm(self.model.parameters(), self.opt['max_grad_norm'])
        # perform updating
        self.optimizer.step()
        # calculate the loss
        loss_val = loss.data[0]
        return loss_val

    def predict(self, inputs):
        # whether moving all data to gpu
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        #target = idx_to_onehot(target, self.opt)
        if self.opt['cuda']:
            inputs = dict([(k, v.cuda()) for k, v in inputs.items()])

        # forward
        self.model.eval()
        logits = self.model.predict(inputs)
        return logits

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)

    # save the model
    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(), # model parameters
                'encoder': self.model.encoder.state_dict(),
                'classifier': self.model.classifier.state_dict(),
                'config': self.opt, # options
                'epoch': epoch # current epoch
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

    # load the model
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Cannot load model from {}".format(filename))
            exit()
        self.model.encoder.load_state_dict(checkpoint['encoder'])
        self.model.classifier.load_state_dict(checkpoint['classifier'])
        self.opt = checkpoint['config']
