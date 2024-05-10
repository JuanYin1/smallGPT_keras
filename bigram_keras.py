import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# hyperparameters
batch_size = 32
block_size = 16
max_iters = 2000
eval_interval = 300
learning_rate = 1e-2
device = 'cpu'  # TensorFlow does not require explicit device specification like PyTorch
eval_iters = 200

np.random.seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = np.array(encode(text), dtype=np.int64)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = np.random.randint(len(data_source) - block_size, size=(batch_size,))
    x = np.stack([data_source[i:i + block_size] for i in ix])
    y = np.stack([data_source[i + 1:i + block_size + 1] for i in ix])
    return x, y


def estimate_loss():
    out = {}
    for split in ['train', 'val']:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss
        out[split] = np.mean(losses)
    return out



class BigramLanguageModel(tf.keras.Model):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = Embedding(vocab_size, vocab_size)

    def call(self, inputs, targets=None):
        logits = self.token_embedding_table(inputs)

        if targets is None:
            loss = None
        else:
            loss = tf.reduce_mean(SparseCategoricalCrossentropy(from_logits=True)(targets, logits))

        return logits, loss

    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(inputs)
            logits = logits[:, -1, :]
            probs = tf.nn.softmax(logits)
            idx_next = tf.random.categorical(probs, num_samples=1, dtype=tf.int32)
            inputs = tf.concat([inputs, idx_next], axis=1)
        return inputs


model = BigramLanguageModel(vocab_size)

optimizer = Adam(learning_rate=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')

    with tf.GradientTape() as tape:
        logits, loss = model(xb, yb)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

context = np.zeros((1, 1), dtype=np.int64)
print(decode(model.generate(context, max_new_tokens=500).numpy().tolist()[0]))
