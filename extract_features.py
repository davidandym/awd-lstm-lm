import argparse as ap
import torch
import torch.nn as nn
import numpy as np

import data
import model

def get_args():
    p = ap.ArgumentParser()
    p.add_argument('--corpus', type=str, 
                   default="data/enwik8/corpus.data")
    p.add_argument('--model', type=str, default='LSTM')
    p.add_argument('--emsize', type=int, default=400)
    p.add_argument('--nhid', type=int, default=1840)
    p.add_argument('--nlayers', type=int, default=3)
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--sentences-file', default=None,
                    help="Sentences input file")
    p.add_argument('--vector-outs', default=None,
                    help="Where to dump sentence representations")
    p.add_argument('--cuda', action='store_false', help='use CUDA')
    return p.parse_args()

def load_sentences(sent_file, c):
    all_sentences = []
    with open(sent_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            symbols = line.split()
            enc_sent = [c.dictionary.word2idx[sym] for sym in symbols]
            all_sentences.append(enc_sent)
    return all_sentences

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

### Main

args = get_args()
fn = args.corpus
corpus = torch.load(fn)
ntokens = len(corpus.dictionary)

## Model 
model = model.RNNModel(args.model, ntokens, args.emsize, 
                       args.nhid, args.nlayers, tie_weights=True)
if args.resume is not None:
    print(f'Loading model from {args.resume}')
    model_load(args.resume)
if args.cuda:
    model = model.cuda()
model.eval()

## Main Loop
all_vectors = []
for sent in load_sentences(args.sentences_file, corpus):
    data = torch.tensor([sent])
    data = data.view(1, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    hidden = model.init_hidden(1)
    model_outs = model(data, hidden, return_h=True)
    result, hidden, raw_outputs, outputs = model_outs
    sent_outputs = raw_outputs[args.nlayers - 1]
    sent_outputs = sent_outputs.cpu().squeeze().detach().numpy()
    all_vectors.append(sent_outputs)

print(f"Extracted features for {len(all_vectors)} sentences.")
print(f"Saving the representations here: {args.vector_outs}")
np.save(args.vector_outs, all_vectors)
