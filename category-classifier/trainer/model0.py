#!/usr/bin/env python

# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/blogs/textclassification


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import re
import tensorflow as tf
import tensorflow.contrib.learn as tflearn
import tensorflow.contrib.layers as tflayers
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
from tensorflow.python.platform import gfile
from tensorflow.contrib import lookup
import logging
import sys
import subprocess
tf.logging.set_verbosity(tf.logging.INFO)

# variables set by init()
TRAIN_STEPS = 1000
EVAL_STEPS = 1000
WORD_VOCAB_FILE = None
N_WORDS = -1

#Vocabulary
MIN_WORD_FREQUENCY=10


# CNN model parameters
EMBEDDING_SIZE = 8
WINDOW_SIZE = EMBEDDING_SIZE
STRIDE = int(WINDOW_SIZE/2)


# describe your data
TARGETS = None
MAX_DOCUMENT_LENGTH = 1000
PADWORD = '[-PAD-]'



TRAIN = None
EVAL = None
logger = None

def init(dataPath, max_samples, outputPath, num_steps,eval_steps):
  global TRAIN_STEPS,EVAL_STEPS, WORD_VOCAB_FILE, N_WORDS,TRAIN,EVAL,logger


  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logger = logging.getLogger('Classifier-Model')

  TRAIN_STEPS = num_steps
  EVAL_STEPS = eval_steps
  WORD_VOCAB_FILE = '{}/vocab.vcb'.format(outputPath)

  data= load_data_and_labels(dataPath);

  logger.info('Targets: %s', TARGETS)

  titles = np.array(data[0]);
  labels = np.array(data[1]);
  p = np.random.permutation(len(titles))
  titles=titles[p]
  labels=labels[p]
  p = None

  EVAL_SIZE= int(len(titles)*0.2);

  N_WORDS = save_vocab(titles, WORD_VOCAB_FILE);
  #N_WORDS = save_vocab('gs://{}/txtcls1/train.csv'.format(BUCKET), 'title', WORD_VOCAB_FILE);

  TRAIN = input(titles[EVAL_SIZE:len(titles)], labels[EVAL_SIZE:len(titles)], 'train')
  EVAL = input(titles[0:EVAL_SIZE], labels[0:EVAL_SIZE], 'eval')
  titles = None
  labels = None



def save_vocab(titles, outfilename):

  logger.info('Building vocabolary...')
  vocab_processor = tflearn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCY)
  vocab_processor.fit(titles)

  with gfile.Open(outfilename, 'wb') as f:
    f.write("{}\n".format(PADWORD))
    for word, index in vocab_processor.vocabulary_._mapping.items():
      f.write("{}\n".format(word))
  nwords = len(vocab_processor.vocabulary_)
  logger.info('Saved {} words into {}'.format(nwords, outfilename))
  return nwords + 2  # PADWORD and <UNK>




def input(data, lables, type):

  if type == 'train':
    mode = tf.contrib.learn.ModeKeys.TRAIN
  else:
    mode = tf.contrib.learn.ModeKeys.EVAL

  # the actual input function passed to TensorFlow
  def _input_fn():
    features = tf.constant(data)
    # make targets numeric
    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(TARGETS), num_oov_buckets=0, default_value=-1)
    target = table.lookup(tf.constant(lables))

    features = {'text': features}
    return features, target

  return _input_fn

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


def load_data_and_labels(dataPath):
    from tensorflow.python.lib.io import file_io
    import os
    global TARGETS;
    # Load data from files
    samplesFiles =subprocess.check_output(['gsutil', 'ls', '{}/*.samples'.format(dataPath)]).splitlines()
    #samplesFiles=['gs://newsriver-category-classifier/data/2.Business-short.samples','gs://newsriver-category-classifier/data/3.Technology-short.samples']
    #samplesFiles=['gs://newsriver-category-classifier/data/2.Business.samples','gs://newsriver-category-classifier/data/3.Technology.samples','gs://newsriver-category-classifier/data/0.International.samples','gs://newsriver-category-classifier/data/5.Sports.samples','gs://newsriver-category-classifier/data/4.Entertainment.samples','gs://newsriver-category-classifier/data/18.United Kingdom.samples','gs://newsriver-category-classifier/data/21.Politics.samples','gs://newsriver-category-classifier/data/49.USA.samples']
    index=0
    allLabels=[];
    allSamples = [];

    TARGETS = ['{}'.format(os.path.basename(file)) for file in samplesFiles]
    for file in samplesFiles:
        logger.info('Loading data file: %s', os.path.basename(file))
        #samples = list(open(file, "r").readlines())
        samples = list(file_io.FileIO(file, mode='r').readlines())

        samples = [clean_str(s.strip()) for s in samples]
        labels = ['{}'.format(os.path.basename(file)) for _ in samples]

        allSamples = allSamples + samples
        allLabels = allLabels + labels;
        index+=1

    return [allSamples, allLabels]




def cnn_model(features, target, mode):
    table = lookup.index_table_from_file(vocabulary_file=WORD_VOCAB_FILE, num_oov_buckets=1, default_value=-1)

    features = features['text']
    # string operations
    logger.info('mode={}'.format(mode))
    words = tf.string_split(features)
    densewords = tf.sparse_tensor_to_dense(words, default_value=PADWORD)
    numbers = table.lookup(densewords)
    padding = tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]])
    padded = tf.pad(numbers, padding)
    sliced = tf.slice(padded, [0,0], [-1, MAX_DOCUMENT_LENGTH])
    logger.info('text={}'.format(sliced))
    logger.info('target={}'.format(target))
    # layer to take the words and convert them into vectors (embeddings)
    embeds = tf.contrib.layers.embed_sequence(sliced, vocab_size=N_WORDS, embed_dim=EMBEDDING_SIZE)


    # now do convolution
    conv = tf.contrib.layers.conv2d(embeds, 1, WINDOW_SIZE, stride=STRIDE, padding='SAME') # (?, 4, 1)
    conv = tf.nn.relu(conv) # (?, 4, 1)
    words = tf.squeeze(conv, [2]) # (?, 4)


    n_classes = len(TARGETS)

    logits = tf.contrib.layers.fully_connected(words, n_classes, activation_fn=None)

    predictions_dict = {
      'source': tf.gather(TARGETS, tf.argmax(logits, 1)),
      'class': tf.argmax(logits, 1),
      'prob': tf.nn.softmax(logits)
    }

    if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
       loss = tf.losses.sparse_softmax_cross_entropy(target, logits)
       train_op = tf.contrib.layers.optimize_loss(
         loss,
         tf.train.get_global_step(),
         optimizer='Adam',
         learning_rate=0.01)
    else:
       loss = None
       train_op = None

    return tflearn.ModelFnOps(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op)


def serving_input_fn():
    feature_placeholders = {
      'text': tf.placeholder(tf.string, [None]),
    }
    features = feature_placeholders
    return tflearn.utils.input_fn_utils.InputFnOps(
      features,
      None,
      feature_placeholders)

def get_train():
    logger.info("Retreiving Train Data")
    return TRAIN

def get_valid():
    logger.info("Retreiving Validation Data")
    return EVAL




from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
def experiment_fn(output_dir):
    # run experiment

    #train_monitors = tf.contrib.learn.monitors.ValidationMonitor(test_set.target, test_set.target,every_n_steps=5)
    #logging_hook = tf.train.LoggingTensorHook({"accuracy" : tflearn.MetricSpec(metric_fn=metrics.streaming_accuracy, prediction_key='class')}, every_n_iter=10)

    return tflearn.Experiment(
        tflearn.Estimator(model_fn=cnn_model, model_dir=output_dir,config=tf.contrib.learn.RunConfig(save_checkpoints_steps=10,save_checkpoints_secs=None,save_summary_steps=100)),
        train_input_fn=get_train(),
        eval_input_fn=get_valid(),
        eval_metrics={
            'acc': tflearn.MetricSpec(
                metric_fn=metrics.streaming_accuracy, prediction_key='class'
            )
        },
        checkpoint_and_export = True,
        train_monitors = None,
        export_strategies=[saved_model_export_utils.make_export_strategy(
            serving_input_fn,
            default_output_alternative_key=None,
            exports_to_keep=1
        )],
        train_steps = TRAIN_STEPS,
        eval_steps = EVAL_STEPS
    )
