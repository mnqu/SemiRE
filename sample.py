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

from model.predictor import Predictor
from model.trainer_predictor import PredictorTrainer
from model.selector import Selector
from model.trainer_selector import SelectorTrainer
from utils import torch_utils, scorer, helper

parser = argparse.ArgumentParser()
parser.add_argument('--p_dir', type=str, help='Directory of the predictor.')
parser.add_argument('--s_dir', type=str, help='Directory of the selector.')
parser.add_argument('--p_model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--s_model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
#parser.add_argument('--dataset', type=str, default='unlabeled', help="Evaluate on dev or test.")
parser.add_argument('--mode', type=str, default='p', help="Sampling mode")
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
model_p_file = args.p_dir + '/' + args.p_model
print("Loading predictor from {}".format(model_p_file))
opt_p = torch_utils.load_config(model_p_file)
predictor = Predictor(opt_p)
model_p = PredictorTrainer(opt_p, predictor)
model_p.load(model_p_file)

model_s_file = args.s_dir + '/' + args.s_model
print("Loading selector from {}".format(model_s_file))
opt_s = torch_utils.load_config(model_s_file)
selector = Predictor(opt_s)
model_s = SelectorTrainer(opt_s, selector)
model_s.load(model_s_file)

# load vocab
TOKEN = data.Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
RELATION = data.Field(sequential=False, unk_token=None, pad_token=None)
POS = data.Field(sequential=True, batch_first=True)
NER = data.Field(sequential=True, batch_first=True)
PST = data.Field(sequential=True, batch_first=True)

fields = {'tokens':('token', TOKEN), 'stanford_pos':('pos', POS), 'stanford_ner':('ner', NER), 'relation':('relation', RELATION), 'subj_pst':('subj_pst', PST), 'obj_pst':('obj_pst', PST)}
dataset_vocab = data.TabularDataset(path=args.data_dir + '/train.json', format='json', fields=fields)
dataset_train = data.TabularDataset(path=args.data_dir + '/labeled.json', format='json', fields=fields)
dataset_unlabeled = data.TabularDataset(path=args.data_dir + '/unlabeled.json', format='json', fields=fields)

TOKEN.build_vocab(dataset_vocab)
RELATION.build_vocab(dataset_vocab)
POS.build_vocab(dataset_vocab)
NER.build_vocab(dataset_vocab)
PST.build_vocab(dataset_vocab)

iterator_train = data.Iterator(dataset=dataset_train, batch_size=50, device=-1, repeat=False, train=True, shuffle=False, sort_key=lambda x: len(x.token), sort_within_batch=True)
iterator_unlabeled = data.Iterator(dataset=dataset_unlabeled, batch_size=50, device=-1, repeat=False, train=True, shuffle=False, sort=True, sort_key=lambda x: -len(x.token), sort_within_batch=False)

#helper.print_config(opt)

nolid = RELATION.vocab.stoi['no_relation']

preds_p, preds_s = [], []
for i, batch in enumerate(iterator_unlabeled):
    inputs = {}
    inputs['words'], inputs['length'] = batch.token
    inputs['pos'] = batch.pos
    inputs['ner'] = batch.ner
    inputs['subj_pst'] = batch.subj_pst
    inputs['obj_pst'] = batch.obj_pst
    inputs['masks'] = torch.eq(batch.token[0], opt_p['vocab_pad_id'])

    target = batch.relation

    if args.cuda:
        inputs = dict([(k, v.cuda()) for k, v in inputs.items()])

    predictor.eval()
    pred = predictor.predict(inputs)
    pred = pred.data.cpu().numpy().tolist()
    preds_p += pred

    selector.eval()
    pred = selector.predict(inputs)
    pred = pred.data.cpu().numpy().tolist()
    preds_s += pred

def arg_max(l):
    bvl, bid = -1, -1
    for k in range(len(l)):
        if l[k] > bvl:
            bvl = l[k]
            bid = k
    return bid, bvl

examples = iterator_unlabeled.data()
num_instance = len(examples)

ranking = list(zip(range(num_instance), preds_p))
ranking = sorted(ranking, key=lambda x:arg_max(x[1])[1], reverse=True)

rel2cn = {}
data_p = []
p, q = 0, 0
for eid, pred in ranking:
    rel, val = arg_max(pred)
    if rel == nolid:
        continue
    data_p += [(eid, rel, val)]
    rel2cn[rel] = rel2cn.get(rel, 0) + 1

    example = examples[eid]
    if RELATION.vocab.itos[rel] == example.relation:
        p += 1
    q += 1

    if q == args.num:
        break

print('P: {} {} {:.2f}'.format(p, q, p * 100 / q))

data_s = []
p, q = 0, 0
for rel in range(len(RELATION.vocab)):
    if rel == nolid:
        continue

    ranking = list(zip(range(num_instance), [preds_s[k][rel] for k in range(num_instance)]))
    ranking = sorted(ranking, key=lambda x:x[1], reverse=True)

    cn = rel2cn.get(rel, 0)

    for k in range(cn):
        eid, val = ranking[k][0], ranking[k][1]
        data_s += [(eid, rel, val)]

        example = examples[eid]
        if RELATION.vocab.itos[rel] == example.relation:
            p += 1
        q += 1

print('S: {} {} {:.2f}'.format(p, q, p * 100 / q))

data = {}
for eid, rel, val in data_p + data_s:
    data[(eid, rel)] = data.get((eid, rel), 0) + 1

p, q = 0, 0
fo = open(args.out, 'w')
for (eid, rel), val in data.items():
    if val != 2:
        continue

    example = examples[eid]
    if RELATION.vocab.itos[rel] == example.relation:
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

print('{} {} {:.2f}'.format(p, q, p * 100 / q))
print("Inference ended.")

