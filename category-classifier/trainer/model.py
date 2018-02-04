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

#Orginal model
#https://github.com/dennybritz/cnn-text-classification-tf

# Nice example  -uses Experiment and custom hookcs
#https://github.com/DongjunLee/text-cnn-tensorflow

#Cool blog post with very high streaming_accuracy
#https://www.google.com/search?q=tensorflow+cnn+text&pws=0&gl=us&source=lnt&tbs=qdr:y&sa=X&ved=0ahUKEwiF6JKJ-v_YAhUEblAKHW64A0YQpwUIHw&biw=1611&bih=930


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
from tensorflow.python.lib.io import file_io
from tensorflow.contrib import lookup
import logging
import sys
import subprocess
import csv
import glob

np.set_printoptions(threshold=np.nan)

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
EMBEDDING_SIZE = 20
NUM_FILTERS = 20
FILTER_SIZES=[5]

#EMBEDDING_SIZE = 128
#NUM_FILTERS = 128
#FILTER_SIZES=[4,5,6]

DROPOUT_KEEP_PROB = 0.5

LEARNING_RATE = 0.001 #0.01

# describe your data
#TARGETS = ['International','Business','Technology','Entertainment','Sports','United Kingdom','Politics','USA']
#TARGETS = ['Business','Technology','Sports','Entertainment','International','United Kingdom','Politics','USA']
#TARGETS = ['Business','Sports']
TARGETS = ['International','Business','Technology','Sports','Entertainment','Italy','Culture','Swiss','TV & Movies','Wine & Dine','Life','Motors','Science','Photography','Health']
LANGUAGE = "it"
NUM_CLASSES = len(TARGETS)

MAX_DOCUMENT_LENGTH = 400
PADWORD = '[-PAD-]'

DATA_PATH = None
logger = None

vocab_processor = tflearn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCY)

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

  N_WORDS = save_vocab(dataPath, WORD_VOCAB_FILE);
  #N_WORDS = save_vocab('gs://{}/txtcls1/train.csv'.format(BUCKET), 'title', WORD_VOCAB_FILE);



def save_vocab(dataPath, outfilename):
    global vocab_processor
    logger.info('Loading text entries for vocabulary...')

    #from numpy import genfromtxt
    #my_data = genfromtxt('/Users/eliapalme/Newsriver/Newsriver-classifier/category-classifier/data/training.csv', delimiter=',')

    if "gs://" not in dataPath:
        samplesFiles = glob.glob('{}/Training-set.{}.csv'.format(dataPath,LANGUAGE))
    else:
        samplesFiles =subprocess.check_output(['gsutil', 'ls', '{}/Training-set.{}.csv'.format(dataPath,LANGUAGE)]).decode("utf-8").splitlines()

    logger.info('Loadinf files for vocabulary: {}'.format(samplesFiles))
    samples = []
    for file in samplesFiles:
        with file_io.FileIO(file, 'r') as csvfile:
            file_samples = [line[2] for line in csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)]
        samples = samples + file_samples;

    logger.info('Building vocabolary...')
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

    if mode == 'training':
        prefix = "Training"
    else:
        prefix = "Evaluation"

    if "gs://" not in dataPath:
        samplesFiles = glob.glob('{}/{}-set.{}.csv'.format(dataPath,prefix,LANGUAGE))
    else:
        samplesFiles =subprocess.check_output(['gsutil', 'ls', '{}/{}-set.{}.csv'.format(dataPath,mode,LANGUAGE)]).decode("utf-8").splitlines()

    #samplesFiles=['gs://newsriver-category-classifier/data/2.Business-short.samples','gs://newsriver-category-classifier/data/3.Technology-short.samples']
    #samplesFiles=['gs://newsriver-category-classifier/data/2.Business.samples','gs://newsriver-category-classifier/data/3.Technology.samples','gs://newsriver-category-classifier/data/0.International.samples','gs://newsriver-category-classifier/data/5.Sports.samples','gs://newsriver-category-classifier/data/4.Entertainment.samples','gs://newsriver-category-classifier/data/18.United Kingdom.samples','gs://newsriver-category-classifier/data/21.Politics.samples','gs://newsriver-category-classifier/data/49.USA.samples']
    #if mode == 'training':
    #    samplesFiles=['/Users/eliapalme/Newsriver/Newsriver-classifier/category-classifier/data/training.csv'];
    #else:
    #    samplesFiles=['/Users/eliapalme/Newsriver/Newsriver-classifier/category-classifier/data/eval.csv'];


    HEADERS = ['language','category','text','title','url','country']

    logger.info('Loadinf files for {}: {}'.format(mode,samplesFiles))

    examples_op = tf.contrib.learn.read_batch_examples(
        samplesFiles,
        batch_size=BATCH_SIZE,
        reader=tf.TextLineReader,
        num_epochs=None,
        parse_fn=lambda x: tf.decode_csv(x, [tf.constant([''], dtype=tf.string)] * len(HEADERS)))

    examples_dict = {}
    for i, header in enumerate(HEADERS):
        examples_dict[header] = examples_op[:, i]

    text_tokens = tf.string_split(examples_dict['text'],delimiter=" ")
    text_tokens = tf.sparse_tensor_to_dense(text_tokens, default_value=PADWORD)

    feature_cols = {'text': text_tokens}

    class_table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(TARGETS), num_oov_buckets=0, default_value=-1)
    classes = class_table.lookup(examples_dict['category'])

    return feature_cols, classes




def cnn_model(features, classes, mode):

    vocab_table = lookup.index_table_from_file(vocabulary_file=WORD_VOCAB_FILE, num_oov_buckets=1, default_value=-1)


    logger.info('mode={}'.format(mode))

    text_tokens = features['text']

    text_ids = vocab_table.lookup(text_tokens)
    text_ids_padded = tf.pad(text_ids, tf.constant([[0,0],[0,MAX_DOCUMENT_LENGTH]]))
    text_ids_padded = tf.slice(text_ids_padded, [0,0], [-1, MAX_DOCUMENT_LENGTH])

     # Keeping track of l2 regularization loss (optional)
    l2_loss = tf.constant(0.0)

    # Embedding layer
    with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedding = tf.get_variable('embedding', [N_WORDS, EMBEDDING_SIZE])
            embedding_inputs = tf.nn.embedding_lookup(embedding, text_ids_padded)
            embedded_chars_expanded = tf.expand_dims(embedding_inputs, -1)

    # Create a convolution + maxpool layer for each filter size
    pooled_outputs = []
    for i, filter_size in enumerate(FILTER_SIZES):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
            # Convolution Layer
            filter_shape = [filter_size, EMBEDDING_SIZE, 1, NUM_FILTERS]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[NUM_FILTERS]), name="b")
            conv = tf.nn.conv2d(
                embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, MAX_DOCUMENT_LENGTH - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool")
            pooled_outputs.append(pooled)

    # Combine all the pooled features
    num_filters_total = NUM_FILTERS * len(FILTER_SIZES)
    h_pool = tf.concat(pooled_outputs, 3)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

    # Add dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, DROPOUT_KEEP_PROB)

    # Final (unnormalized) scores and predictions
    with tf.name_scope("output"):
        W = tf.get_variable(
            "W",
            shape=[num_filters_total, NUM_CLASSES],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        logits = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
        predictions = tf.argmax(logits, 1, name="predictions")

    # Accuracy
    #with tf.name_scope("accuracy"):
    #    correct_predictions = tf.equal(predictions, tf.argmax(target, 1))
    #    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


    predictions_dict = {
      'source': tf.gather(TARGETS, tf.argmax(logits, 1)),
      'class': tf.argmax(logits, 1),
      'prob': tf.nn.softmax(logits)
    }

    if mode == tf.contrib.learn.ModeKeys.TRAIN or mode == tf.contrib.learn.ModeKeys.EVAL:
       loss = tf.losses.sparse_softmax_cross_entropy(classes, logits)
       train_op = tf.contrib.layers.optimize_loss(
         loss,
         tf.train.get_global_step(),
         optimizer='Adam',
         learning_rate=LEARNING_RATE)
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
      'text': tf.placeholder(tf.string, shape=[None,None]),
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
