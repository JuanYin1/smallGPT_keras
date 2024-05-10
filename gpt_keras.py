import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import keras as k
from tensorflow.keras import layers, Model
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



class Head(k.Model): 
    def __init__(self, head_size): 
        super().__init__()
        self.key = k.layers.Dense(n_embd, head_size, use_bias=False)
        self.query = k.layers.Dense(n_embd, head_size, use_bias=False)
        self.value = k.layers.Dense(n_embd, head_size, use_bias=False)
        
        self.dropout = k.layers.Dropout(dropout)

    def call(self, x): 
        B, T, C = x.shape
        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('inf')) # (B, T, T)
        wei = k.activations.softmax(wei) # (B, T, T)
        wei = self.dropout(wei)

        # perform weighted aggregation of the values
        v = self.value(x)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out 


class MultiHeadAttention(k.Model): 
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size): 
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.proj = k.layers.Dense(n_embd, n_embd)
        self.dropout = k.layers.Dropout(dropout)

    def call(self, x): 
        out = k.layers.concatenate([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(k.Layer):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd): 
        super().__init__()
        self.l1 = k.layers.Dense(n_embd, 4 * n_embd)
        self.relu = k.layers.Activation('relu')
        self.l2 = k.layers.Dense(4 * n_embd, n_embd)
        self.dropout = k.layers.Dropout(dropout)
    
    def call(self, x): 
        x = self.relu(self.l1(n_embd))
        x = self.dropout(self.l2(x))
        return x
    
class LayerNorm(k.Layer):
    def __init__(self, ndim, bias):
        super(LayerNorm, self).__init__()
        self.weight = self.add_weight(shape=(ndim,),
                                      initializer='ones',
                                      trainable=True,
                                      name='weight')
        if bias:
            self.bias = self.add_weight(shape=(ndim,),
                                        initializer='zeros',
                                        trainable=True,
                                        name='bias')
        else:
            self.bias = None

    def call(self, input):
        return tf.nn.layer_norm(input, center=self.bias is not None,
                                 scale=True,
                                 gamma=self.weight,
                                 beta=self.bias,
                                 variance_epsilon=1e-5)


class Block(layers.Layer):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super(Block, self).__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = layers.LayerNormalization()
        self.ln2 = layers.LayerNormalization()

    def call(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(Model):

    def __init__(self, vocab_size, n_embd, block_size, n_layer):
        super(GPTLanguageModel, self).__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = layers.Embedding(vocab_size, n_embd)
        self.position_embedding_table = layers.Embedding(block_size, n_embd)
        self.blocks = tf.keras.Sequential([Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = layers.LayerNormalization() # final layer norm
        self.lm_head = layers.Dense(vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self._init_weights()

    def _init_weights(self):
        for layer in self.layers:
            if isinstance(layer, layers.Dense):
                layer.kernel_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
                if layer.bias is not None:
                    layer.bias_initializer = tf.keras.initializers.Zeros()
            elif isinstance(layer, layers.Embedding):
                layer.embeddings_initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    def call(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(tf.range(T, dtype=tf.int32)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = tf.reshape(logits, (B*T, C))
            targets = tf.reshape(targets, (B*T,))
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, logits, from_logits=True))

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = tf.nn.softmax(logits, axis=-1) # (B, C)
            # sample from the distribution
            idx_next = tf.random.categorical(logits, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = tf.concat((idx, idx_next), axis=1) # (B, T+1)
        return idx

model = GPTLanguageModel()
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = Adam(learning_rate=learning_rate)

for iter in range(max_iters):

    # Every once in a while, evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    with tf.device('/GPU:0'):  # Use GPU if available
        logits, loss = model(xb, yb)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

context = tf.zeros((1, 1), dtype=tf.int32)
generated_text = decode(model.generate(context, max_new_tokens=500)[0].numpy().tolist())
print(generated_text)
