# Project part 3 and 4
#Import libraries
import time
import os
import sys
from six.moves.urllib.request import urlretrieve
import tarfile
import numpy as np
import pandas as pd
import pickle
from scipy import misc
from tqdm import tqdm
import matplotlib.pyplot as plt
%matplotlib inline
import cv2
import math
import sys
from imgaug import augmenters as iaa
from datetime import datetime
import os
import seaborn as sns
import csv
import imageio
import tensorflow.compat.v1 as tf1
tf1.disable_v2_behavior()
import tensorflow
import logging
import h5py
import keras
from tensorflow import keras
from nltk.corpus import stopwords
import sklearn.manifold
import gensim
import string
from random import randint
from keras.layers import Embedding
from keras.models import Sequential
import tensorflow as tf
import glob
import multiprocessing
!pip install sentence_transformers
from __future__ import absolute_import, division, print_function
from sentence_transformers import SentenceTransformer
import scipy.spatial
import seaborn as sns
import nltk as nltk
import re
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import random
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import array
from pickle import dump

nltk.download('punkt')
nltk.download('stopwords')

# Project part 3 

# Data set download
url = 'https://www.cs.toronto.edu/~kriz/'
last_percent_reported = None

#-----------------------------------------------------------------------------------------------------------------------
# download cifar100

def download_progress_hook(count, blockSize, totalSize):
    """
    A hook to report the progress of a download. This is mostly intended for users with slow internet connections. 
    Reports every 5% change in download progress. 
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent


def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    dest_filename = os.path.join('.', filename)
    if force or not os.path.exists(dest_filename):
        print('Attempting to download:', filename)
        filename, _ = urlretrieve(url + filename, dest_filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(dest_filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', dest_filename)
    else:
        raise Exception(
            'Failed to verify ' + dest_filename + '. Can you get to it with a browser?')
    return dest_filename
maybe_download('cifar-100-python.tar.gz', 169001437)

def maybe_extract(filename, force=False):
    '''
    Unzip the downloaded file
    '''
    root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
    if os.path.isdir(root) and not force:
        # You may override by setting force=True.
        print('%s already present - Skipping extraction of %s.' % (root, filename))
    else:
        print('Extracting data for %s. This may take a while. Please wait.' % root)
        tar = tarfile.open(filename)
        sys.stdout.flush()
        tar.extractall('.')
        tar.close()


dataset = os.path.join('.', 'cifar-100-python.tar.gz')
maybe_extract(dataset)
#-----------------------------------------------------------------------------------------------------------------------
# Get cifar100
def unpickle(file):
    '''
    Pickle loading CIFAR-100 data
    '''
    with open(file, 'rb') as fo:
        res = pickle.load(fo, encoding='bytes')
    return res

# Meta, Train, and Test are dictionaries containing data
meta = unpickle('cifar-100-python/meta')
fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]
train = unpickle('cifar-100-python/train')
test = unpickle('cifar-100-python/test')
filenames = [t.decode('utf8') for t in train[b'filenames']]
fine_labels = train[b'fine_labels']
data = train[b'data']

images = list()
for d in data:
    image = np.zeros((32,32,3), dtype=np.uint8)
    image[...,0] = np.reshape(d[:1024], (32,32)) # Red channel
    image[...,1] = np.reshape(d[1024:2048], (32,32)) # Green channel
    image[...,2] = np.reshape(d[2048:], (32,32)) # Blue channel
    images.append(image)


with open('cifar100.csv', 'w+') as f:
    for index,image in tqdm(enumerate(images)):
        filename = filenames[index]
        label = fine_labels[index]
        label = fine_label_names[label]

        imageio.imsave('cifar-100-python/img%s' %filename, image)

        f.write('cifar-100-python/img%s,%s\n'%(filename,label))

#Classes
Classes = pd.DataFrame(meta[b'fine_label_names'],columns = ['Classes'])
data = data.reshape(50000, 3, 32, 32).transpose(0,2,3,1).astype("uint8")

#-----------------------------------------------------------------------------------------------------------------------
#**** Data Augmentation*****
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.CropAndPad(px=(-2, 2),sample_independently=True,pad_mode=["constant", "edge"]),
    iaa.Affine(shear=(-10, 10),mode = ['symmetric','wrap']),
    iaa.Add((-5, 5)),
    iaa.Multiply((0.8, 1.2)),
],random_order=True)

#Applying data augmentation to dataset
all_train, all_labels = [], []
for index in range(6):
    all_train.extend(seq.augment_images(data)/255)
    all_labels.extend(train[b'fine_labels'])

#-----------------------------------------------------------------------------------------------------------------------
# shuffle the data to randomize any pattern in the order of the dataset.
all_train_shuffled, all_labels_shuffled = [], []
combined = list(zip(all_train, all_labels))
random.shuffle(combined)
all_train_shuffled[:], all_labels_shuffled[:] = zip(*combined)
num_class = 100
all_train_shuffled = np.asarray(all_train_shuffled)
train_len = len(all_train_shuffled)

# create one-hot vectors
def one_hot(vec, vals=num_class):
    n = len(vec)
    out = np.zeros((n, vals))
    out[range(n), vec] = 1
    return out

all_labels_shuffled= one_hot(all_labels_shuffled, num_class)

test_shuffled = np.vstack(test[b"data"])
test_len = len(test_shuffled)

test_shuffled = test_shuffled.reshape(test_len,3,32,32).transpose(0,2,3,1)/255
test_labels = one_hot(test[b'fine_labels'], num_class)

#-----------------------------------------------------------------------------------------------------------------------
class CifarHelper():
    '''
    feeding the data into the model one small batch at a time. 
    '''
    def __init__(self):
        self.i = 0

        self.training_images = all_train_shuffled
        self.training_labels = all_labels_shuffled

        self.test_images = test_shuffled
        self.test_labels = test_labels

    def next_batch(self, batch_size=100):
        x = self.training_images[self.i:self.i + batch_size].reshape(100, 32, 32, 3)
        y = self.training_labels[self.i:self.i + batch_size]
        self.i = (self.i + batch_size) % len(self.training_images)
        return x, y

ch = CifarHelper()
x = tf1.placeholder(tf1.float32,shape=[None,32,32,3])
y_true = tf1.placeholder(tf1.float32,shape=[None,num_class])
hold_prob = tf1.placeholder(tf1.float32)


#Functions for initializing layers
def init_weights(shape):
    return tf1.Variable(tf1.truncated_normal(shape, stddev=0.1))

def init_bias(shape):
    return tf1.Variable(tf1.constant(0.1, shape=shape))

def conv2d(x, W):
    return tf1.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2by2(x):
    return tf1.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

def convolutional_layer(input_x, shape):
    return tf1.nn.relu(conv2d(input_x, init_weights(shape)) + init_bias([shape[3]]))

def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    return tf1.matmul(input_layer, init_weights([input_size, size])) + init_bias([size])

def get_model(x):
    convo_1 = convolutional_layer(x,shape=[3,3,3,32])
    convo_2 = convolutional_layer(convo_1,shape=[3,3,32,64])
    convo_2_pooling = max_pool_2by2(convo_2)
    convo_3 = convolutional_layer(convo_2_pooling,shape=[3,3,64,128])
    convo_4 = convolutional_layer(convo_3,shape=[3,3,128,256])
    convo_4_pooling = max_pool_2by2(convo_4)
    convo_2_flat = tf1.reshape(convo_4_pooling,[-1,8*8*256])
    full_layer_one = tf1.nn.relu(normal_full_layer(convo_2_flat,1024))
    full_one_dropout = tf1.nn.dropout(full_layer_one,hold_prob)
    y_pred = normal_full_layer(full_one_dropout,100)
    return y_pred

y_pred= get_model(x)

#Loss Function
softmaxx = tf1.nn.softmax_cross_entropy_with_logits_v2(labels = y_true,logits = y_pred)
cross_entropy = tf1.reduce_mean(softmaxx)
# This is optimizer
optimizer = tf1.train.AdamOptimizer(.001)
train = optimizer.minimize(cross_entropy)
init = tf1.global_variables_initializer()

saver = tf1.train.Saver()


#Running the model
print(str(datetime.now()) + '\n')
minibatch_check, accuracy  = 500, 0
accuracy_list = []
target_accuracy = 1
with tf1.Session(config=config) as sess:
    sess.run(init)
    i = 0
    while (accuracy < target_accuracy):
        i = i + 1

        batch = ch.next_batch(100)

        sess.run(train, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 0.5})

        if i %500== 0:
            print("step: {}".format(i))
            matches = tf1.equal(tf1.argmax(y_pred, 1), tf1.argmax(y_true, 1))

            acc = tf1.reduce_mean(tf1.cast(matches, tf1.float32))

            print('Train Accuracy:')
            print(sess.run(acc, feed_dict={x: batch[0], y_true: batch[1], hold_prob: 1.0}))

            # new
            batch_accuracy = []
            for k in range(0, int(len(test_shuffled) / minibatch_check)):
                batch_accuracy.append(
                    sess.run(acc, feed_dict={x: test_shuffled[minibatch_check * (k):minibatch_check * (k + 1)],
                                             y_true: test_labels[minibatch_check * (k):minibatch_check * (k + 1)],
                                             hold_prob: 1.0}))
            print('Test Accuracy:')
            accuracy = sum(batch_accuracy) / (len(batch_accuracy))
            print(accuracy)
            accuracy_list.append(accuracy)
            print('\n')

        if (accuracy > target_accuracy):
            saver.save(sess, 'cifar-100-python/models/model53.ckpt')
        plt.plot(accuracy_list)
        
#Restore the model
model_path = 'cifar-100-python/models/model53.ckpt'
print(accuracy_list)

#-----------------------------------------------------------------------------------------------------------------------
# Restore the model
cuts, predictions = 200, []
with tf1.Session() as sess:
    
    saver.restore(sess,model_path)

    probabilities = tf1.nn.softmax(y_pred)
    matches2 = softmaxx
    acc2 = tf1.cast(probabilities,tf1.float32)
    for k in range(0,int(len(test_shuffled)/cuts)):
        predictions.extend(sess.run(acc2,feed_dict={x:test_shuffled[cuts*(k):cuts*(k+1)], y_true:test_labels[cuts*(k):cuts*(k+1)], hold_prob:1.0}))
    predictions = np.array(predictions)
#-----------------------------------------------------------------------------------------------------------------------
# save all data
# Add the model to the CSV file
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array(all_train_shuffled, dtype=np.float32)
ys = np.array(predictions, dtype=np.float32)
model.save('image_model.h5')


myarray = np.fromfile('image_model.h5')
print(myarray)

#Save predictions
output = pd.DataFrame(predictions)
output.to_csv('predictions.csv')

predictions_df = np.argmax(predictions,1)
predictions_df = pd.DataFrame(predictions_df)
predictions_df.to_csv('predictions_df.csv')

test_labels_df = np.argmax(test_labels,1)
test_labels_df = pd.DataFrame(test_labels_df)
test_labels_df.to_csv('test_labels.csv')

Classes = pd.DataFrame(Classes)
Classes.to_csv('Classes.csv')

#-----------------------------------------------------------------------------------------------------------------------
#Compare Actual Value with Predicted Value
img_num = 250
plt.figure(figsize=(.6,.6))
plt.xticks([])
plt.yticks([])
plt.imshow(test_shuffled[img_num])

labels_not_vec = np.argmax(test_labels,1)
Classes.iloc[labels_not_vec[img_num]]
print('True Label: '+str(Classes.iloc[labels_not_vec[img_num]]).split()[1])
print('Prediction: '+str(Classes.iloc[predictions_df.iloc[img_num]]).split()[2])



def show_compare(test_shuffled, df, number, title):
    f, axarr = plt.subplots(3,5)
    img_nums = np.random.randint(0,len(test_shuffled),3*5)
    for i in range(0,3):
        for j in range(0,5):
            axarr[i,j].imshow(test_shuffled[img_nums[(i*5)+j]])
            axarr[i,j].set_title(str(Classes.iloc[df[img_nums[(i+1)*(j+1)-1]]]).split()[number])
            axarr[i,j].axis('off')
            f.suptitle(title)
    
show_compare(test_shuffled, labels_not_vec, 1, 'Actual Values')
show_compare(test_shuffled, predictions_df, 2, 'Predicted Values')

#-----------------------------------------------------------------------------------------------------------------------        
def plot_prediction(image):
    '''
    Helper Function
    '''
     try_out, predictions = [], []
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized_image = cv2.resize(img, (32, 32)) 
    try_out.append(resized_image/255)


    with tf1.Session() as sess:
        saver.restore(sess,model_path)
        probabilities = tf1.nn.softmax(y_pred)
        matches2 = softmaxx
        acc2 = tf1.cast(probabilities,tf1.float32)
        predictions.extend(sess.run(acc2,feed_dict={x:try_out,hold_prob:1.0}))
        predictions = np.array(predictions)
        predictions_df = pd.DataFrame(predictions).T
        predictions_df = predictions_df.sort_values(0,ascending=0)
        predictions_df = predictions_df[:10].T
        predictions_df.columns = Classes.iloc[predictions_df.columns.values]
        columns = predictions_df.columns
        columns_list = []
        for i in range(len(columns)):
            columns_list.append(str(columns[i])[3:-3])
        columns_list = pd.DataFrame(columns_list)
        predictions_df.columns = pd.DataFrame(columns_list)
        predictions_df = predictions_df.T
        predictions_df.columns=['Probability']
        predictions_df['Prediction'] = predictions_df.index
        f, axarr = plt.subplots(1,2, figsize=(10,4))
        axarr[0].imshow(img)
        axarr[0].axis('off')
        axarr[1] = sns.barplot(x="Probability", y="Prediction", data=predictions_df,color="red",)
        sns.set_style(style='white')
        axarr[1].set_ylabel('')    
        axarr[1].set_xlabel('')
        axarr[1].grid(False)
        axarr[1].spines["top"].set_visible(False)
        axarr[1].spines["right"].set_visible(False)
        axarr[1].spines["bottom"].set_visible(False)
        axarr[1].spines["left"].set_visible(False)
        plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        f.suptitle("Model Prediction")
        f.subplots_adjust(top=0.88)

plot_prediction('dog.jpg')


#-----------------------------------------------------------------------------------------------------------------------
# Project part 4

# read BBC news dataset file
dataCsv=pd.read_csv('BBC news dataset.csv')
dFrame = pd.DataFrame(dataCsv)
dFrame = dFrame.dropna(subset=['description','tags'])

#Preprocessing
def Preprocessing(dFrame)
    # convert to lowercase
    dFrame['description']=dFrame['description'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    # remove special characters
    dFrame['description']=dFrame['description'].str.replace('\d+', '').str.replace('[^\w\s]', '')
    # tokenize
    dFrame['description'] = dFrame['description'].apply(nltk.word_tokenize)
    # stemming
    dFrame['description'] = dFrame['description'].apply(lambda x: [nltk.stem.PorterStemmer().stem(y) for y in x])
    return dFrame

dFrame= Preprocessing(dFrame)
listDataChars, tokens= [] , []
for x in dFrame['description']:
  listDataChars.append(list(x))

for el in listDataChars:
  tokens.extend(el)
tokens=(list(dict.fromkeys(tokens)))

# organize into sequences of tokens
sequence_lines=[]
lables=[]
for i in range(51,len(tokens)):
  seq=list(tokens[i-51:i])
  line=' '.join(seq)
  sequence_lines.append(line)
  lables.append([''.join(tokens[i])])

# integer encode sequences of words
tokenizer=Tokenizer()
tokenizer.fit_on_texts(sequence_lines)
sequences=tokenizer.texts_to_sequences(sequence_lines)
sequences=np.array(sequences)
# vocabulary size
vocab_size=len(tokenizer.word_index)+1
# separate into input and output
x1,y1=sequences[:,:-1],sequences[:,-1]
y1=to_categorical(y1,num_classes=vocab_size)
seq_length=x1.shape[1]

def get_model_headline(seq_length,vocab_size)
    # define model LSTM
    modle_= Sequential()
    modle_.add(Embedding(vocab_size,50,input_length=seq_length))
    modle_.add(LSTM(300))
    modle_.add(Dense(300,activation='relu'))
    modle_.add(Dense(vocab_size,activation='softmax'))
    # compile model
    modle_.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return modle_

modle_=get_model_headline(seq_length,vocab_size)
# fit model
modle_.fit(x1,y1,batch_size=256,epochs=100)
# save the model to file
modle_.save('text_model.h5')
# save the tokenizer
dump(tokenizer, open('tokenizer.pkl', 'wb'))

# select a seed text
seed_text =sequence_lines[randint(0,len(sequence_lines))]

# generate a sequence from a language model
def generate_test_seq(modle, tokenizer, test_seq_length, seed_text,n_words):
  text=[]
  for _ in range(n_words):
    encoded= tokenizer.texts_to_sequences([seed_text])[0]
    encoded=pad_sequences([encoded],maxlen=test_seq_length, truncating='pre')

    y_predict=modle.predict_classes(encoded)
    predicted_word=''
    for word, index in tokenizer.word_index.items():
      if index==y_predict:
        predicted_word=word
        break
    seed_text=seed_text+' '+predicted_word
    text.append(predicted_word)
  return ' '.join(text)

Keywords=[]
# generate new text
Keywords.append(top_name)
Keywords.append(generate_test_seq(modle, tokenizer, seq_length, Keywords[0], 1))
Keywords.append(generate_test_seq(modle, tokenizer, seq_length, Keywords[1], 1))

print("Keywords is: ")
print(Keywords)
