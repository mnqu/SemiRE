import os
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchtext import data

from model.predictor import Predictor
from model.trainer_predictor import PredictorTrainer
from utils import scorer, helper

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='train.json', help="Evaluate on dev or test.")
parser.add_argument('--emb_dim', type=int, default=300, help='Word embedding dimension.')
parser.add_argument('--ner_dim', type=int, default=30, help='NER embedding dimension.')
parser.add_argument('--pos_dim', type=int, default=30, help='POS embedding dimension.')
parser.add_argument('--hidden_dim', type=int, default=200, help='RNN hidden state size.')
parser.add_argument('--num_layers', type=int, default=2, help='Num of RNN layers.')
parser.add_argument('--dropout', type=float, default=0.5, help='Input and RNN dropout rate.')

parser.add_argument('--attn', dest='attn', action='store_true', help='Use attention layer.')
parser.add_argument('--no-attn', dest='attn', action='store_false')
parser.set_defaults(attn=True)
parser.add_argument('--attn_dim', type=int, default=200, help='Attention size.')
parser.add_argument('--pe_dim', type=int, default=30, help='Position encoding dimension.')

parser.add_argument('--lr', type=float, default=1.0, help='Applies to SGD and Adagrad.')
parser.add_argument('--lr_decay', type=float, default=0.9)
parser.add_argument('--optim', type=str, default='sgd', help='sgd, adagrad, adam or adamax.')
parser.add_argument('--num_epoch', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
parser.add_argument('--log_step', type=int, default=20, help='Print log every k steps.')
parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# make opt
opt = vars(args)

# load data
print("Loading data from {} with batch size {}...".format(opt['data_dir'], opt['batch_size']))
TOKEN = data.Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
RELATION = data.Field(sequential=False, unk_token=None, pad_token=None)
POS = data.Field(sequential=True, batch_first=True)
NER = data.Field(sequential=True, batch_first=True)
PST = data.Field(sequential=True, batch_first=True)

fields = {'tokens':('token', TOKEN), 'stanford_pos':('pos', POS), 'stanford_ner':('ner', NER), 'relation':('relation', RELATION), 'subj_pst':('subj_pst', PST), 'obj_pst':('obj_pst', PST)}
dataset_vocab = data.TabularDataset(path=opt['data_dir'] + '/train.json', format='json', fields=fields)
dataset_train = data.TabularDataset(path=opt['data_dir'] + '/' + opt['dataset'], format='json', fields=fields)
dataset_dev = data.TabularDataset(path=opt['data_dir'] + '/dev.json', format='json', fields=fields)

TOKEN.build_vocab(dataset_vocab)
RELATION.build_vocab(dataset_vocab)
POS.build_vocab(dataset_vocab)
NER.build_vocab(dataset_vocab)
PST.build_vocab(dataset_vocab)

iterator_train = data.Iterator(dataset=dataset_train, batch_size=opt['batch_size'], device=-1, repeat=False, train=True, shuffle=True, sort_key=lambda x: len(x.token), sort_within_batch=True)
iterator_dev = data.Iterator(dataset=dataset_dev, batch_size=opt['batch_size'], device=-1, repeat=False, train=True, sort_key=lambda x: len(x.token), sort_within_batch=True)

opt['num_class'] = len(RELATION.vocab)
opt['vocab_pad_id'] = TOKEN.vocab.stoi['<pad>']
opt['pos_pad_id'] = POS.vocab.stoi['<pad>']
opt['ner_pad_id'] = NER.vocab.stoi['<pad>']
opt['pe_pad_id'] = PST.vocab.stoi['<pad>']
opt['vocab_size'] = len(TOKEN.vocab)
opt['pos_size'] = len(POS.vocab)
opt['ner_size'] = len(NER.vocab)
opt['pe_size'] = len(PST.vocab)

TOKEN.vocab.load_vectors('glove.840B.300d')
if TOKEN.vocab.vectors is not None:
    opt['emb_dim'] = TOKEN.vocab.vectors.size(1)

# prepare for model saving
model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
model_save_dir = opt['save_dir'] + '/' + model_id
opt['model_save_dir'] = model_save_dir
helper.ensure_dir(model_save_dir, verbose=True)

# save config
helper.save_config(opt, model_save_dir + '/config.json', verbose=True)
# logger for training information
file_logger = helper.FileLogger(model_save_dir + '/' + opt['log'], header="# epoch\ttrain_loss\tdev_loss\tdev_f1")

# print model info
helper.print_config(opt)

# model
predictor = Predictor(opt, emb_matrix=TOKEN.vocab.vectors)
model = PredictorTrainer(opt, predictor)

dev_f1_history = []
current_lr = opt['lr']

global_step = 0
global_start_time = time.time()
format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
max_steps = len(iterator_train) * opt['num_epoch']

# start training
for epoch in range(1, opt['num_epoch']+1):
    train_loss = 0
    for i, batch in enumerate(iterator_train):
        start_time = time.time()
        global_step += 1

        inputs = {}
        inputs['words'], inputs['length'] = batch.token
        inputs['pos'] = batch.pos
        inputs['ner'] = batch.ner
        inputs['subj_pst'] = batch.subj_pst
        inputs['obj_pst'] = batch.obj_pst
        inputs['masks'] = torch.eq(batch.token[0], opt['vocab_pad_id'])

        target = batch.relation

        loss = model.update(inputs, target)
        train_loss += loss
        if global_step % opt['log_step'] == 0:
            duration = time.time() - start_time
            print(format_str.format(datetime.now(), global_step, max_steps, epoch,\
                    opt['num_epoch'], loss, duration, current_lr))

    # eval on dev
    print("Evaluating on dev set...")
    predictions = []
    golds = []
    dev_loss = 0
    for i, batch in enumerate(iterator_dev):
        inputs = {}
        inputs['words'], inputs['length'] = batch.token
        inputs['pos'] = batch.pos
        inputs['ner'] = batch.ner
        inputs['subj_pst'] = batch.subj_pst
        inputs['obj_pst'] = batch.obj_pst
        inputs['masks'] = torch.eq(batch.token[0], opt['vocab_pad_id'])

        target = batch.relation

        preds, _, loss = model.predict(inputs, target)
        predictions += preds
        dev_loss += loss
        golds += target.data.tolist()
    predictions = [RELATION.vocab.itos[p] for p in predictions]
    golds = [RELATION.vocab.itos[p] for p in golds]
    dev_p, dev_r, dev_f1 = scorer.score(golds, predictions)
    
    # print training information
    train_loss = train_loss / len(iterator_train) * opt['batch_size'] # avg loss per batch
    dev_loss = dev_loss / len(iterator_dev) * opt['batch_size']
    print("epoch {}: train_loss = {:.6f}, dev_loss = {:.6f}, dev_f1 = {:.4f}".format(epoch,\
            train_loss, dev_loss, dev_f1))
    file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}".format(epoch, train_loss, dev_loss, dev_f1))

    # save the current model
    model_file = model_save_dir + '/checkpoint_epoch_{}.pt'.format(epoch)
    model.save(model_file, epoch)
    if epoch == 1 or dev_f1 > max(dev_f1_history):
        copyfile(model_file, model_save_dir + '/best_model.pt')
        print("new best model saved.")
    if epoch % opt['save_epoch'] != 0:
        os.remove(model_file)
    
    # lr schedule
    # change learning rate
    if len(dev_f1_history) > 10 and dev_f1 <= dev_f1_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad']:
        current_lr *= opt['lr_decay']
        model.update_lr(current_lr)

    dev_f1_history += [dev_f1]
    print("")

print("Training ended with {} epochs.".format(epoch))

