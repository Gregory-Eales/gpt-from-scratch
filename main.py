import numpy as np
import torch

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


class AttentionHead(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        # how to determine the dims
        self.q = torch.nn.Linear()
        self.k = torch.nn.Linear()
        self.v = torch.nn.Linear()

        self.linear = torch.nn.Linear()

    def forward(self, x):

        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        scores = torch.softmax( q.dot(k) / torch.sqrt(x.shape[0]))

        out = v.dot(scores)

        self.linear(out)

        return out


class Block():
    
    # is this the full encoder?

    def __init__(self):
        pass

if __name__ == "__main__":
    pass
