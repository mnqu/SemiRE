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

# the relation model
class PredictorTrainer(object):
    """ A wrapper class for the training and evaluation of models. """
    def __init__(self, opt, predictor):
        # options
        self.opt = opt
        # encoding model
        self.model = predictor
        # loss function
        self.criterion = nn.CrossEntropyLoss()
        # all parameters of the model
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        # whether moving all data to gpu
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        # intialize the optimizer
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
    
    # train the model with a batch
    def update(self, inputs, target):
        """ Run a step of forward and backward model update. """
        # whether moving all data to gpu
        # inputs: [feature size, batch size, seq length]
        # labels: [batch size, seq length]
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
        loss = self.criterion(logits, target)
        
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

    def predict(self, inputs, target):
        # whether moving all data to gpu
        """ Run forward prediction. If unsort is True, recover the original order of the batch. """
        if self.opt['cuda']:
            inputs = dict([(k, v.cuda()) for k, v in inputs.items()])
            target = target.cuda()

        # forward
        self.model.eval()
        logits, _ = self.model(inputs)
        loss = self.criterion(logits, target)
        probs = F.softmax(logits).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        return predictions, probs, loss.data[0]

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
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']
