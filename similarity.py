Project part 2
import nltk as nltk
import re
import pandas as pd
from nltk.corpus import stopwords
import sklearn.manifold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import gensim.downloader as api
import gensim
import string
import io
import os
import keras
import sys
from keras.layers import Embedding
from keras.models import Sequential
import numpy as np
import tensorflow as tf
import gensim.models.word2vec as w2v
import glob
import multiprocessing
#!pip install sentence_transformers
from __future__ import absolute_import, division, print_function
from sentence_transformers import SentenceTransformer
import scipy.spatial
import seaborn as sns
nltk.download('punkt')
nltk.download('stopwords')

# get data from file BBC news dataset
dataCsv=pd.read_csv('BBC news dataset.csv')
dFrame = pd.DataFrame(dataCsv)
dFrame = dFrame.dropna(subset=['description','tags'])
cols=['description','tags']

#Preprocessing
for y in cols:
    # convert to lowercase
    dFrame[y]=dFrame[y].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # remove special characters
    dFrame[y]=dFrame[y].str.replace('\d+', '').str.replace('[^\w\s]', '')
    # tokenize
    dFrame[y] = dFrame[y].apply(nltk.word_tokenize)
    # remove stopwords
    dFrame[y]=dFrame[y].apply(lambda x: [item for item in x if item not in stopwords.words('english')])
    # stemming
    dFrame[y] = dFrame[y].apply(lambda x: [nltk.stem.PorterStemmer().stem(y) for y in x])

# one hot vector
i=3
listOneHot=[]
for y in cols:
    for x in dFrame[y]:
        integer_encoded = LabelEncoder().fit_transform(x)
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        listOneHot.append(onehot_encoder.fit_transform(integer_encoded))
    dFrame.insert(i, y+'OneHot', listOneHot, True)
    i+=1
    listOneHot=[]


for y in cols:
    # joining with " "
    dFrame[y] = dFrame[y].str.join(" ")


corpus =list(dFrame['description'])
embedder = SentenceTransformer('bert-base-nli-mean-tokens')
corpus_embeddings = embedder.encode(corpus)
# Query sentences:
queries =list(dFrame['tags'])
query_embeddings = embedder.encode(queries)
closest_n = 1
mostSimilar=[]
for query, query_embedding in zip(queries, query_embeddings):
    distances = scipy.spatial.distance.cdist([query_embedding], corpus_embeddings, "cosine")[0]

    results = zip(range(len(distances)), distances)
    results = sorted(results, key=lambda x: x[1])

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 1 most similar sentences in corpus:")

    for idx, distance in results[0:closest_n]:
        mostSimilar.append(corpus[idx].strip())
        print(corpus[idx].strip(), "(Score: %.4f)" % (1-distance))

with open("query.txt", "w") as outfile:
    outfile.write("\n".join(queries))


listQueries=[]
listMostSimilar=[]
for x in queries:
  listQueries.append(x.split())
for x in mostSimilar:
    listMostSimilar.append(x.split())

# ***Creating Word embedding***

num_features=300
# Minimum word count threshold
min_word_count=3
num_workers=multiprocessing.cpu_count()
context_size=7
# Downsample setting for frequent words
downsampling=1e-3
size=1
# Seed for the RNG, to make the results reproducible
#random number generator
thrones2vec = w2v.Word2Vec(
    sg=1,
    seed=1,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample=downsampling
)

thrones2vec.build_vocab(listQueries)
print(len(thrones2vec.wv.vocab))

thrones2vec.train(listQueries, total_words=len(thrones2vec.wv.vocab), epochs=100)

if not os.path.exists("train"):
  os.makedirs("train")

thrones2vec.save(os.path.join("train","thrones2vac.w2v"))
thrones2vec=w2v.Word2Vec.load(os.path.join("train","thrones2vac.w2v"))
tsne = sklearn.manifold.TSNE(n_components=2, random_state=0)
all_word_vectors_matrix=thrones2vec.wv.syn0

# Set parameters
vocab_size=len(thrones2vec.wv.vocab)
max_length=len(thrones2vec.wv.vocab)
# Generate random embedding matrix for sake of illustration
embedding_matrix = np.random.rand(vocab_size,300)
print(all_word_vectors_matrix)
model = Sequential()
model.add(Embedding(vocab_size, 300, weights=[all_word_vectors_matrix],
input_length=max_length, trainable=False))
# Average the output of the Embedding layer over the word dimension
model.add(keras.layers.Lambda(lambda x: keras.backend.mean(x, axis=1)))
model.summary()
model.save("modle.h5")

all_word_vectors_matrix_2d= tsne.fit_transform(all_word_vectors_matrix)

points=pd.DataFrame(
    [
      (word, coords[0], coords[1] )
      for word, coords in [
                    (word, all_word_vectors_matrix_2d[thrones2vec.wv.vocab[word].index])
                     for word in thrones2vec.wv.vocab
      ]
    ],
    columns=["word","x","y"]
)

sns.set_context("poster")
points.plot.scatter("x", "y", s=50, figsize=(20,12))

myarray = np.fromfile('modle.h5')
print(myarray)


