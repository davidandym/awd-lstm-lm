import os
import random
import torch

from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Shard(object):
    def __init__(self, ids):
        self.shard = torch.tensor(ids, dtype=torch.long)

class Corpus(object):
    def __init__(self, path, shard_dir=None, byte_voc=False):
        self.dictionary = Dictionary()
        # self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.tokenize(os.path.join(path, 'train.txt'), save=False, byte_voc=byte_voc)
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'), byte_voc=byte_voc)
        self.test = self.tokenize(os.path.join(path, 'test.txt'), byte_voc=byte_voc)

        shard_count = 0
        self.train_shards = []

        for shard in self.shard_train(os.path.join(path, 'train.txt')):
            f = 'corpus.train-{}.data'.format(shard_count)
            fn = os.path.join(shard_dir, f)
            print('saved shard {}'.format(fn))
            self.train_shards.append(fn)
            torch.save(shard, fn)
            shard_count += 1

    def shard_train(self, path, avg_shard_size=1000000000):
        """ Shard a large training file into multiple small datasets. """
        with open(path, 'r') as f:
            shard_encoded = []
            for line in f:
                toks = line.strip().split()
                ids = [self.dictionary.word2idx[tok] for tok in toks]
                shard_encoded += ids
                if len(shard_encoded) >= avg_shard_size:
                    yield shard_encoded
                    shard_encoded = []
            yield shard_encoded

    def iterate_train_shards(self):
        shuffled = random.sample(self.train_shards, k=len(self.train_shards))           
        for shard in shuffled:
            s = torch.load(shard)
            yield s.shard

    def tokenize(self, path, byte_voc=False, save=True):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        if byte_voc:
            for i in range(256):
                self.dictionary.add_word(str(i))

        if not byte_voc:
            with open(path, 'r') as f:
                tokens = 0
                for line in f:
                    words = line.strip().split()
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)
        elif save:
            with open(path, 'r') as f:
                tokens = 0
                for line in f:
                    words = line.strip().split()
                    tokens += len(words)


        # Tokenize file content
        if save:
            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    words = line.split()
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1

            return ids
