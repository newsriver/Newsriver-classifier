# -*- coding: utf-8 -*-
"""
Simple example using a Dynamic RNN (LSTM) to classify IMDB sentiment dataset.
Dynamic computation are performed over sequences with variable length.
References:
    - Long Short Term Memory, Sepp Hochreiter & Jurgen Schmidhuber, Neural
    Computation 9(8): 1735-1780, 1997.
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
Links:
    - http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
    - http://ai.stanford.edu/~amaas/data/sentiment/
"""

#sudo pip3 install tflearn
#


from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
import numpy as np
import glob
import re

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(train=True):

    # Load data from files
    path='./data/train/*.samples'
    samplesFiles = glob.glob(path)
    #samplesFiles=['/Users/eliapalme/Newsriver/Newsriver-classifier/training/data/train/2.Business.samples','/Users/eliapalme/Newsriver/Newsriver-classifier/training/data/train/3.Technology.samples']
    index=0
    allLabels=[];
    allSamples = [];
    for file in samplesFiles:
        print("Loading data file: {}".format(file))
        samples = list(open(file, "r").readlines())
        #samples = samples[:1000]
        samples = [clean_str(s.strip()) for s in samples]
        labels = [index for _ in samples]

        allSamples = allSamples + samples
        allLabels = allLabels + labels;
        index+=1

    return [allSamples, allLabels]




data= load_data_and_labels();

titles = np.array(data[0]);
labels = np.array(data[1]);

print("Randomizing data")
p = np.random.permutation(len(titles))
titles=titles[p]
labels=labels[p]


MAX_DOCUMENT_LENGTH = 100
NUM_OF_CLASSES = 9

print("Building Vocabulary")
vocabulary  = tflearn.data_utils.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=0, vocabulary=None, tokenizer_fn=None)
vocabulary = vocabulary.fit(titles);
words = np.array(list(vocabulary.transform(titles)));


half = int(len(words)/2);

trainX = words[0:half]
trainY = to_categorical(labels[0:half], nb_classes=NUM_OF_CLASSES)

testX = words[half:len(words)]
testY = to_categorical(labels[half:len(words)], nb_classes=NUM_OF_CLASSES)

print(trainX)
print(trainY);


net = tflearn.input_data([None, MAX_DOCUMENT_LENGTH])
val = len(vocabulary.vocabulary_)
net = tflearn.embedding(net, input_dim=val+2, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, NUM_OF_CLASSES, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy')
print(trainX);
print(trainY);

print("Traning")
# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,batch_size=32) # I know validation and training are same here







# Data preprocessing
# NOTE: Padding is required for dimension consistency. This will pad sequences
# with 0 at the end, until it reaches the max sequence length. 0 is used as a
# masking value by dynamic RNNs in TFLearn; a sequence length will be
# retrieved by counting non zero elements in a sequence. Then dynamic RNN step
# computation is performed according to that length.
#trainX = pad_sequences(trainX, maxlen=100, value=0.)
#testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
#trainY = to_categorical(trainY,10)
#testY = to_categorical(testY,10)

# Network building
#net = tflearn.input_data([None, 100])
# Masking is not required for embedding, sequence length is computed prior to
# the embedding op and assigned as 'seq_length' attribute to the returned Tensor.
#net = tflearn.embedding(net, input_dim=10000, output_dim=128)
#net = tflearn.lstm(net, 128, dropout=0.8, dynamic=True)
#net = tflearn.fully_connected(net, 2, activation='softmax')
#net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy')

# Training
#model = tflearn.DNN(net, tensorboard_verbose=0)
#model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
#          batch_size=32)
