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

#https://github.com/gaussic/text-classification-cnn-rnn
#http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/
#https://github.com/gaussic/text-classification-cnn-rnn/blob/master/cnn_model.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
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
import csv
tf.logging.set_verbosity(tf.logging.INFO)

# variables set by init()
TRAIN_STEPS = 1000
EVAL_STEPS = 1000
BATCH_SIZE =  64
WORD_VOCAB_FILE = None
N_WORDS = -1
CHECKPOINT_STEPS = 1000
SUMMARY_STEPS = 100
#Vocabulary
MIN_WORD_FREQUENCY=10


# CNN model parameters
EMBEDDING_SIZE = 24
WINDOW_SIZE = EMBEDDING_SIZE
STRIDE = int(WINDOW_SIZE/2)


# describe your data
#TARGETS = ['International','Business','Technology','Entertainment','Sports','United Kingdom','Politics','USA']

TARGETS = ['Business','Technology']


MAX_DOCUMENT_LENGTH = 500
PADWORD = '[-PAD-]'

DATA_PATH = None
logger = None

def init(dataPath, outputPath, num_steps,eval_steps):
  global TRAIN_STEPS,EVAL_STEPS, WORD_VOCAB_FILE, N_WORDS,logger,DATA_PATH


  logging.basicConfig(stream=sys.stdout, level=logging.INFO)
  logger = logging.getLogger('Classifier-Model')

  TRAIN_STEPS = num_steps
  EVAL_STEPS = eval_steps
  DATA_PATH = dataPath
  WORD_VOCAB_FILE = '{}/vocab.vcb'.format(outputPath)

  logger.info('Targets: %s', TARGETS)

  texts,labels = load_data_and_labels(DATA_PATH,'training')

  N_WORDS = save_vocab(texts, WORD_VOCAB_FILE);
  #N_WORDS = save_vocab('gs://{}/txtcls1/train.csv'.format(BUCKET), 'title', WORD_VOCAB_FILE);



def save_vocab(path, outfilename):

    logger.info('Loading text entries for vocabulary...')

    #from numpy import genfromtxt
    #my_data = genfromtxt('/Users/eliapalme/Newsriver/Newsriver-classifier/category-classifier/data/training.csv', delimiter=',')

    with open('/Users/eliapalme/Newsriver/Newsriver-classifier/category-classifier/data/training.csv', 'r') as csvfile:
        samples = [line[2] for line in csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)]

    logger.info('Building vocabolary...')
    vocab_processor = tflearn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCY)
    vocab_processor.fit(samples)


    with gfile.Open(outfilename, 'wb') as f:
        f.write("{}\n".format(PADWORD))
        for word, index in vocab_processor.vocabulary_._mapping.items():
            f.write("{}\n".format(word))

    nwords = len(vocab_processor.vocabulary_)
    logger.info('Saved {} words into {}'.format(nwords, outfilename))
    return nwords + 2  # PADWORD and <UNK>



def load_data_and_labels(dataPath,mode):
    from tensorflow.python.lib.io import file_io
    import os
    # Load data from files
    #samplesFiles =subprocess.check_output(['gsutil', 'ls', '{}/{}*.csv'.format(dataPath,mode)]).splitlines()
    #samplesFiles=['gs://newsriver-category-classifier/data/2.Business-short.samples','gs://newsriver-category-classifier/data/3.Technology-short.samples']
    #samplesFiles=['gs://newsriver-category-classifier/data/2.Business.samples','gs://newsriver-category-classifier/data/3.Technology.samples','gs://newsriver-category-classifier/data/0.International.samples','gs://newsriver-category-classifier/data/5.Sports.samples','gs://newsriver-category-classifier/data/4.Entertainment.samples','gs://newsriver-category-classifier/data/18.United Kingdom.samples','gs://newsriver-category-classifier/data/21.Politics.samples','gs://newsriver-category-classifier/data/49.USA.samples']
    if mode == 'training':
        samplesFiles=['/Users/eliapalme/Newsriver/Newsriver-classifier/category-classifier/data/training.csv'];
    else:
        samplesFiles=['/Users/eliapalme/Newsriver/Newsriver-classifier/category-classifier/data/eval.csv'];

    index=0
    allLabels=[];
    allSamples = [];

    HEADERS = ['id','category','text']


    logger.info('Loadinf files: {}'.format(samplesFiles))

    examples_op = tf.contrib.learn.read_batch_examples(
        samplesFiles,
        batch_size=BATCH_SIZE,
        reader=tf.TextLineReader,
        num_epochs=None,
        parse_fn=lambda x: tf.decode_csv(x, [tf.constant([''], dtype=tf.string)] * len(HEADERS)))

    examples_dict = {}
    for i, header in enumerate(HEADERS):
        examples_dict[header] = examples_op[:, i]

    feature_cols = {'text': examples_dict['text']}
    #feature_cols.update({'text': dense_to_sparse( examples_dict['text'])})

    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(TARGETS), num_oov_buckets=0, default_value=-1)

    labels = table.lookup(examples_dict['category'])
    #label= tf.string_to_number(examples_dict['id'], out_type=tf.int32)

    return feature_cols, labels





def cnn_model(features, target, mode):


    embedding_dim = 64  # 词向量维度
    seq_length = 600  # 序列长度
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 128  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 64  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard



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
    #logger.info('text={}'.format(sliced.eval()))
    #logger.info('target={}'.format(target.eval()))


    with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [N_WORDS, EMBEDDING_SIZE])
            embedding_inputs = tf.nn.embedding_lookup(embedding, sliced)

    with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, num_filters, kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

    with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, hidden_dim, name='fc1')
            fc = tf.contrib.layers.dropout(fc, dropout_keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            logits = tf.layers.dense(fc, len(TARGETS), name='fc2')


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
    def _input_fn():
        return load_data_and_labels(DATA_PATH,'training')
    return _input_fn

def get_valid():
    logger.info("Retreiving Validation Data")
    def _input_fn():
        return load_data_and_labels(DATA_PATH,'eval')
    return _input_fn




from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
def experiment_fn(output_dir):
    # run experiment

    #train_monitors = tf.contrib.learn.monitors.ValidationMonitor(test_set.target, test_set.target,every_n_steps=5)
    #logging_hook = tf.train.LoggingTensorHook({"accuracy" : tflearn.MetricSpec(metric_fn=metrics.streaming_accuracy, prediction_key='class')}, every_n_iter=10)

    return tflearn.Experiment(
        tflearn.Estimator(model_fn=cnn_model, model_dir=output_dir,config=tf.contrib.learn.RunConfig(save_checkpoints_steps=CHECKPOINT_STEPS,save_checkpoints_secs=None,save_summary_steps=SUMMARY_STEPS)),
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
