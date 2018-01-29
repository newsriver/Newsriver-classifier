#!/usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import re
import logging
import sys
import subprocess
import csv

logger = logging.getLogger('Converter')

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


def load_data_and_labels():
    from tensorflow.python.lib.io import file_io
    import os
    global TARGETS;
    # Load data from files
    #samplesFiles =subprocess.check_output(['gsutil', 'ls', '{}/*.samples'.format('gs://newsriver-category-classifier/data')]).splitlines()
    #samplesFiles=['gs://newsriver-category-classifier/data/2.Business.samples']
    #samplesFiles=['gs://newsriver-category-classifier/data/2.Business.samples','gs://newsriver-category-classifier/data/3.Technology.samples','gs://newsriver-category-classifier/data/0.International.samples','gs://newsriver-category-classifier/data/5.Sports.samples','gs://newsriver-category-classifier/data/4.Entertainment.samples','gs://newsriver-category-classifier/data/18.United Kingdom.samples','gs://newsriver-category-classifier/data/21.Politics.samples','gs://newsriver-category-classifier/data/49.USA.samples']

    samplesFiles=['/Users/eliapalme/Newsriver/Newsriver-classifier/category-classifier/data/2.Business.samples','/Users/eliapalme/Newsriver/Newsriver-classifier/category-classifier/data/3.Technology.samples']

    allLabels=[];
    allSamples = [];

    TARGETS = ['{}'.format(os.path.basename(file)) for file in samplesFiles]
    for file in samplesFiles:
        logger.info('Loading data file: %s', os.path.basename(file))
        #samples = list(open(file, "r").readlines())
        #samples = list(file_io.FileIO(file, mode='r').readlines())

        samples = np.array([clean_str(s.strip()) for s in file_io.FileIO(file, mode='r').readlines()])
        labels = np.array(['{}'.format(os.path.basename(file)) for _ in samples])

        allSamples = np.concatenate([allSamples,samples])
        allLabels = np.concatenate([allLabels,labels])

    return [allSamples, allLabels]

texts,labels = load_data_and_labels()
with open('training.csv', 'w') as f_t:
    with open('eval.csv', 'w') as f_e:
        writer_training = csv.writer(f_t,delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        writer_eval = csv.writer(f_e,delimiter=',', quoting=csv.QUOTE_NONNUMERIC)

        for i in range(0, len(texts)):
            if i%5==0:
                writer_eval.writerow([int(labels[i].split(".")[0]),labels[i].split(".")[1],texts[i]])
            else:
                writer_training.writerow([int(labels[i].split(".")[0]),labels[i].split(".")[1],texts[i]])
