import numpy as np
import torch

from torch.nn import functional as F

import math

import torch.nn as nn
torch.manual_seed(1337)

# load dataset
text = open('data/data.txt', 'r').read()
chars = sorted(list(set(text)))
print(chars)
    
# set and print the vocab size
vocab_size = len(chars)
print("text has a vocab size of {} characters".format(vocab_size))

# hash map for conversion
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}

# define encode / decode functions
encode = lambda s: [stoi[c] for c in s] # string -> lists of nums
decode = lambda l: "".join([itos[i] for i in l]) # list of nums -> string

# encode the data and convert it to a torch tensor
encoded_text = encode(text)
data = torch.Tensor(encoded_text)

# get the test and training data
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]
print('data tensor has a shape of {}'.format(data.shape[0]))
print('train and test data have shapes {} and {} respectively'.format(len(train_data), len(test_data)))

def load_dataset(path='data/data.txt'):
    text = open(path, 'r').read()


class Config(object):

    def __init__(self):
        self.N


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn_dropout = nn.Dropout(config.attn_dropout)

        self.res_dropout = nn.Dropout(config.res_dropout)

        # not sure how this works yet...
        self.register_buffer("bias", torch.tril(
            torch.ones(
                config.block_size,
                config.block_size
            )).view(1, 1, config.block_size, config.block_size)
        )

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v 
        y = y.transpose(1, 2).contiguous().view(B, T, C) 

        y = self.res_dropout(self.c_proj(y))
        return y



class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.l1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.l2 = nn.LayerNorm(config.n_embd)


 if __name__ == "__main__":
    pass
