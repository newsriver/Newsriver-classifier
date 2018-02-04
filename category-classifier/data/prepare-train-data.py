#!/usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import re
import sys
import subprocess
import csv
import glob
from random import shuffle

language = "it"

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9()]", " ", string)

    #string = re.sub(r"\'s", " \'s", string)
    #string = re.sub(r"\'ve", " \'ve", string)
    #string = re.sub(r"n\'t", " n\'t", string)
    #string = re.sub(r"\'re", " \'re", string)
    #string = re.sub(r"\'d", " \'d", string)
    #string = re.sub(r"\'ll", " \'ll", string)
    #string = re.sub(r",", " , ", string)
    #string = re.sub(r"!", " ! ", string)
    #string = re.sub(r"\(", " \( ", string)
    #string = re.sub(r"\)", " \) ", string)
    #string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)


    """
    Instead of reducing the sentences to a list of tokens (>=3 chars) we could sonsider to keep
    the sencence structure (see commented experssion above) and even avoid to lowe case all chars
    """

    string = string.strip().lower()

    sentence = ""
    words = string.split(" ")
    for word in words:
        if len(word) >= 3:
            sentence += word + " "

    return sentence.strip()


def load_data_and_labels():
    from tensorflow.python.lib.io import file_io
    import os
    global TARGETS;



    samplesFiles = glob.glob('{}/{}.*.csv'.format(".",language))

    data=[];

    for filename in samplesFiles:
        print('Loading data file: {}'.format(os.path.basename(filename)))
        with open(filename, 'r') as csvfile:
            csvlines = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            for row in csvlines:
                data.append([os.path.basename(filename).split(".")[0],os.path.basename(filename).split(".")[1],clean_str(row[3]),clean_str(row[1]),row[2],row[4]])

    return data


data = load_data_and_labels()
print('Randomizing samples...')
shuffle(data)


print('Writing training and evaluation sets...')
with open('Training-set.{}.csv'.format(language), 'w',encoding='utf8') as f_t:
    with open('Evaluation-set.{}.csv'.format(language), 'w',encoding='utf8') as f_e:
        writer_training = csv.writer(f_t,delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        writer_eval = csv.writer(f_e,delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

        for i in range(0, len(data)):
            if i%5==0:
                writer_eval.writerow(data[i])
            else:
                writer_training.writerow(data[i])
