# inspired by https://www.youtube.com/watch?v=kCc8FmEb1nY

import torch
import torch.nn as nn
import math
from torch.nn import functional as F
import time

torch.manual_seed(1337)

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
        device = 'cpu'
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
        device = 'cpu'
else:
    print("MPS is available on this system.")
    device = 'mps'
device = 'cpu' # for some reason macbook pro M1 Max runtime on cpu is 6x faster than GPU/mps
datatype = torch.long # cross entropy loss doesn't work with int32
# hyperparameters
batch_size = 32
block_size = 8
learning_rate = 1e-2
max_iters = 3000
eval_iters = 300
eval_interval = 300
# !wget  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

print("length of data set in characters: ", len(text))
print("Sample document from input \n--------\n", text[:100])
print("---------")
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("vocabulary ",''.join(chars))
print("vocab_size: ",vocab_size)

# create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
# encoder : take a string return a list of integer
encode = lambda s : [stoi[c] for c in s]
#decoder : take a list of integer and return a string
decode = lambda l: ''.join([itos[i] for i in l])

print("result of encoder: ", encode('hi there'))
print("result of decoder: ", decode(encode('hi there')))


data = torch.tensor(encode(text), dtype=datatype)
print(f"input data shape: {data.shape}, input data type: {data.dtype}")
#print(data[:100])

# lets separate the dataset into train and test set
partition_point = int(0.9*data.shape[0])
train_data = data[:partition_point]
val_data = data[partition_point:]

def get_batch(split, block_size, batch_size):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size - 2, (batch_size,)) # Adding -2 to prevent y to roll over to beginning of data
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+1+block_size] for i in ix])
  x,y = x.to(device), y.to(device)
  return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval() # dropout will behave differently during eval or training
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split, batch_size, block_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # create a embedding table of dimension vocab_size
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        # the value in kth dimension of embedding matrix will be the probability/logit of kth word being the next word
        # Hence we need to have hidden_size=vocab_size

    def forward(self, idx, targets = None):
        # Size idx (B,T) B=Batch,T=Time,C=Channel (no of hidden vector)
        # ideally we should keep the same orientation of logits for both cases which isn't the case here

        logits = self.token_embedding_table(idx) # (B,T,C)
        if targets == None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) # expects (B,C,T)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits,_ = self(idx) #(B,T,C) here passing the context is meaningless but for generalization sake
            logits = logits[:,-1,:] #(B,C) only pick the last output
            prob = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(prob, num_samples=1) #(B,1) sample the most probable token, adding temperature and high T will improve diversity
            idx = torch.cat((idx, idx_next), dim=1) #(B,T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)


optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

start_time = time.time()
for iter in range(max_iters):
    xb, yb = get_batch('train', batch_size, block_size)

    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % eval_interval  == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.3f}, val loss {losses['val']:.3f}")

print("Training finished in time: ", time.time() - start_time)


context = torch.zeros((1,1), dtype=datatype, device=device)
print(decode(m.generate(context, max_new_tokens=400)[0].tolist()))
