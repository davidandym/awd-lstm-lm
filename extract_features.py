import argparse as ap
import torch
import torch.nn as nn

import model

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.dropouth, args.dropouti, args.dropoute, args.wdrop, args.tied)

# load and eval mode to disable dropout etc.
model_load(args.resume)
model.eval()

fn = 'corpus.data'
fn = os.path.join('', fn)

corpus = torch.load(fn)

sent = bytes('David is not happy.', encoding='utf-8')
enc_sent = [corpus.dictionary.word2idx[word] for word in sent]

print(enc_sent)

hidden = model.init_hidden(batch_size)
model_outs = model(enc_sent, hidden, return_h=True)

result, hidden, raw_outputs, outputs = model_outs

# there should be no difference here
print(raw_outputs)
print(outputs)
