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
from utils import torch_utils, scorer, helper

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='unlabeled', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")
parser.add_argument('--thresh', type=float, default=0.8)

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
predictor = Predictor(opt)
model = PredictorTrainer(opt, predictor)
model.load(model_file)

# load vocab
TOKEN = data.Field(sequential=True, batch_first=True, lower=True, include_lengths=True)
RELATION = data.Field(sequential=False, unk_token=None, pad_token=None)
POS = data.Field(sequential=True, batch_first=True)
NER = data.Field(sequential=True, batch_first=True)
PST = data.Field(sequential=True, batch_first=True)

fields = {'tokens':('token', TOKEN), 'stanford_pos':('pos', POS), 'stanford_ner':('ner', NER), 'relation':('relation', RELATION), 'subj_pst':('subj_pst', PST), 'obj_pst':('obj_pst', PST)}
dataset_vocab = data.TabularDataset(path=opt['data_dir'] + '/train.json', format='json', fields=fields)
dataset_train = data.TabularDataset(path=opt['data_dir'] + '/labeled.json', format='json', fields=fields)
dataset_unlabeled = data.TabularDataset(path=opt['data_dir'] + '/unlabeled.json', format='json', fields=fields)

TOKEN.build_vocab(dataset_vocab)
RELATION.build_vocab(dataset_vocab)
POS.build_vocab(dataset_vocab)
NER.build_vocab(dataset_vocab)
PST.build_vocab(dataset_vocab)

iterator_train = data.Iterator(dataset=dataset_train, batch_size=opt['batch_size'], device=-1, repeat=False, train=True, shuffle=False, sort_key=lambda x: len(x.token), sort_within_batch=True)
iterator_unlabeled = data.Iterator(dataset=dataset_unlabeled, batch_size=opt['batch_size'], device=-1, repeat=False, train=True, shuffle=False, sort=True, sort_key=lambda x: -len(x.token), sort_within_batch=False)

helper.print_config(opt)

predictions = []
for i, batch in enumerate(iterator_unlabeled):
    inputs = {}
    inputs['words'], inputs['length'] = batch.token
    inputs['pos'] = batch.pos
    inputs['ner'] = batch.ner
    inputs['subj_pst'] = batch.subj_pst
    inputs['obj_pst'] = batch.obj_pst
    inputs['masks'] = torch.eq(batch.token[0], opt['vocab_pad_id'])

    target = batch.relation

    if opt['cuda']:
        inputs = dict([(k, v.cuda()) for k, v in inputs.items()])
    predictor.eval()
    preds, _ = predictor(inputs)
    preds = torch.nn.functional.softmax(preds)
    preds = preds.data.cpu().numpy().tolist()

    predictions += preds

num_instance = len(predictions)

def arg_max(l):
    bvl, bid = -1, -1
    for k in range(len(l)):
        if l[k] > bvl:
            bvl = l[k]
            bid = k
    return bid, bvl

p, q = 0, 0

examples = iterator_unlabeled.data()

fo = open(args.out, 'w')
for k in range(num_instance):
    bid, bvl = arg_max(predictions[k])

    if RELATION.vocab.itos[bid] == 'no_relation' or bvl < args.thresh:
        continue

    example = examples[k]
    if RELATION.vocab.itos[bid] == example.relation:
        p += 1
    q += 1

    output = {}
    output['tokens'] = example.token
    output['stanford_pos'] = example.pos
    output['stanford_ner'] = example.ner
    output['subj_pst'] = example.subj_pst
    output['obj_pst'] = example.obj_pst
    output['relation'] = RELATION.vocab.itos[bid]
    fo.write(json.dumps(output) + '\n')
fo.close()    

print('{} {} {:.2f}'.format(p, q, p * 100 / q))
print("Inference ended.")

