# -*- coding: utf-8 -*-
"""question1_mca.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1s9CWNkBdT_pwpq4efD9ds404ys_HUFIL
"""

import nltk
nltk.download('abc')
nltk.download('punkt')

"""#### Skip gram model is used for making word embeddings."""

from nltk.corpus import abc
from nltk.tokenize import RegexpTokenizer
import torch
from tqdm import tqdm

'''
  The size of the corpus is : 663964
  The Vocabulary size is : 11557
'''

cut_indx = 70000
corp = abc.raw()
wds1 = corp.split()[:cut_indx]
print(len(wds1))
t = 1e-5
# this is the frequency
d = dict()
for i in wds1:
  d[i] = 0
for i in wds1:
  d[i] += 1

wds = list()
for j in wds1:
  if (d[j] >= 5):
    wds.append(j)

# updated frequency
freq = {}
for i in wds:
  freq[i] = 0
for i in wds:
  freq[i] += 1
# Generate vocab
vocab = set(wds)
probs = {}

# Probabilities for subsampling
for i in wds:
  probs[i] = (1 - (t/freq[i])**(0.5))

# loading the words. these words are in sequence as they appear in the raw sentence
# print(vocab)
w2i = {w:i for i, w in enumerate(vocab)}
i2w = {i:w for i, w in enumerate(vocab)}
vocab_size = len(vocab)
print(vocab_size)

"""Skip gram model with window size = 2"""

# dataset creation
import random

def generate_sample():
  print("generating samples")
  num = random.random()
  incl = list()

  for i in wds:
    if (num <= probs[i]):
      # accept
      incl.append(i)

  wds_sampled = list()
  for i in wds:
    if i in incl:
      wds_sampled.append(i)

  wd_sz = 2
  data = list()

  # positive samples
  for i in range(wd_sz, len(wds_sampled) - wd_sz):
    for j in range(-1*wd_sz, wd_sz + 1):
      if (j != 0):
        data.append((wds_sampled[i], wds_sampled[i+j], 1))
    
    # negative samples
    ns = 0
    while (ns != 4):
      nm = random.randint(0, len(wds_sampled)-wd_sz)
      if (i - wd_sz <= nm <= i + wd_sz):
        continue
      else:
        ns += 1
        data.append((wds_sampled[i], wds_sampled[nm], 0))

  return data

# print(generate_sample()[0])

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class network(nn.Module):
  def __init__(self, v_size, emb_size):
    super(network, self).__init__()
    self.embeddings = nn.Embedding(v_size, emb_size).cuda()

  def forward(self, focus, context):
      embed_focus = self.embeddings(focus).view((1, -1))
      embed_ctx = self.embeddings(context).view((1, -1))
      score = torch.mm(embed_focus, torch.t(embed_ctx))
      log_probs = F.logsigmoid(score)

      return log_probs

embd_size = 50
learning_rate = 0.0001
n_epoch = 20

losses = []
loss_fn = nn.MSELoss()
model = network(vocab_size, embd_size)
model.cuda()
loss_fn.cuda()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
data = generate_sample()
print("Size of the dataset : " + str(len(data)))
print("starting to train")

embeddings = list()
lses = list()

for epoch in range(n_epoch):
  total_loss = .0
  for in_w, out_w, target in tqdm(data):
    in_w_var = Variable(torch.LongTensor([w2i[in_w]])).cuda()
    out_w_var = Variable(torch.LongTensor([w2i[out_w]])).cuda()
    
    optimizer.zero_grad()
    log_probs = model(in_w_var, out_w_var)
    loss = loss_fn(log_probs[0], Variable(torch.Tensor([target])).cuda())
    
    loss.backward()
    optimizer.step()

    total_loss += loss.item()

  embeddings.append(model.embeddings.weight)
  lses.append(total_loss)

torch.save(model.state_dict(), '/content/drive/My Drive/word2vec.pt')

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import numpy as np

# %matplotlib inline

def tsne_plot(label, embedding):
  print('Plotting...')
  plt.figure(figsize=(16, 9))
  colors = cm.rainbow(np.linspace(0, 1, 1))
  x = embedding[:,0]
  y = embedding[:,1]
  plt.scatter(x, y, c=colors, alpha=0.2, label=label)
  plt.legend(loc=4)
  plt.grid(True)
  plt.show()

for i, e in enumerate(embeddings):
  tsne_ak_2d = TSNE(perplexity=30, n_components=2, init='pca', n_iter=3500, random_state=32)
  e = e.detach().cpu().numpy()
  embeddings_ak_2d = tsne_ak_2d.fit_transform(e)
  tsne_plot('Epoch #' + str(i+1), embeddings_ak_2d)