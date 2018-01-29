import numpy as np
import re
import itertools
import glob
from collections import Counter


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
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    path='./data/'
    if train:
        path+= 'train/*.samples'
    else:
        path+= 'cross/*.samples'

    samplesFiles = glob.glob(path)
    index=0
    allLabels=None;
    allSamples = [];
    labelsMap = {};
    for file in samplesFiles:
        lable = [0] * len(samplesFiles)
        lable[index]=1
        labelsMap[index]=file.split('.')[2]
        index+=1

        samples = list(open(file, "r").readlines())
        samples = [clean_str(s.strip()) for s in samples]

        labels = [lable for _ in samples]

        allSamples = allSamples + samples
        if allLabels is None :
            allLabels=labels
        else:
            allLabels = np.concatenate([allLabels, labels], 0)

    return [allSamples, allLabels, labelsMap]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
