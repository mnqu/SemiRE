import os
import random
import argparse
import pickle
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchtext import data

from model.selector import Selector
from model.trainer_selector import SelectorTrainer
from utils import torch_utils, scorer, helper

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='unlabeled', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")
parser.add_argument('--num', type=int, default=1000)

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
selector = Selector(opt)
model = SelectorTrainer(opt, selector)
model.load(model_file)

# load vocab
TOKEN = data.Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
RELATION = data.Field(sequential=False, unk_token=None, pad_token=None)
POS = data.Field(sequential=True, batch_first=True)
NER = data.Field(sequential=True, batch_first=True)
PST = data.Field(sequential=True, batch_first=True)

fields = {'tokens':('token', TOKEN), 'stanford_pos':('pos', POS), 'stanford_ner':('ner', NER), 'relation':('relation', RELATION), 'subj_pst':('subj_pst', PST), 'obj_pst':('obj_pst', PST)}
dataset_vocab = data.TabularDataset(path=opt['data_dir'] + '/train.json', format='json', fields=fields)
dataset_unlabeled = data.TabularDataset(path=opt['data_dir'] + '/unlabeled.json', format='json', fields=fields)

TOKEN.build_vocab(dataset_vocab)
RELATION.build_vocab(dataset_vocab)
POS.build_vocab(dataset_vocab)
NER.build_vocab(dataset_vocab)
PST.build_vocab(dataset_vocab)

iterator_unlabeled = data.Iterator(dataset=dataset_unlabeled, batch_size=opt['batch_size'], device=-1, repeat=False, train=True, shuffle=False, sort=True, sort_key=lambda x: -len(x.token), sort_within_batch=False)

helper.print_config(opt)

preds = []
for i, batch in enumerate(iterator_unlabeled):
    inputs = {}
    inputs['words'], inputs['length'] = batch.token
    inputs['pos'] = batch.pos
    inputs['ner'] = batch.ner
    inputs['subj_pst'] = batch.subj_pst
    inputs['obj_pst'] = batch.obj_pst
    inputs['masks'] = torch.eq(batch.token[0], opt['vocab_pad_id'])

    pred = torch.nn.functional.sigmoid(model.predict(inputs))
    pred = pred.data.cpu().numpy().tolist()

    preds += pred

num_instance = len(preds)
examples = iterator_unlabeled.data()
nolid = RELATION.vocab.stoi['no_relation']

ranking = {}
for eid in range(num_instance):
    for rel in range(opt['num_class']):
        if rel == nolid:
            continue
        ranking[(eid, rel)] = preds[eid][rel]

ranking = sorted(ranking.items(), key=lambda x:x[1], reverse=True)

p, q = 0, 0

fo = open(args.out, 'w')
for (eid, rel), val in ranking:
    example = examples[eid]
    if rel == RELATION.vocab.stoi[example.relation]:
        p += 1
    q += 1

    output = {}
    output['tokens'] = example.token
    output['stanford_pos'] = example.pos
    output['stanford_ner'] = example.ner
    output['subj_pst'] = example.subj_pst
    output['obj_pst'] = example.obj_pst
    output['relation'] = RELATION.vocab.itos[rel]
    fo.write(json.dumps(output) + '\n')

    if q == args.num:
        break
fo.close()

print('{} {} {:.2f}'.format(p, q, p * 100 / q))
print("Inference ended.")

