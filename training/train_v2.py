#https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/learn/text_classification_cnn.py
#https://towardsdatascience.com/how-to-do-text-classification-using-tensorflow-word-embeddings-and-cnn-edae13b3e575


import tensorflow as tf
from tensorflow.contrib import lookup
import tensorflow.contrib.learn as tflearn
from tensorflow.contrib.learn.python.learn import learn_runner
import tensorflow.contrib.metrics as metrics
from tensorflow.python.platform import gfile
from tensorflow.contrib.learn.python.learn.utils import saved_model_export_utils
import numpy as np


EMBEDDING_SIZE = 2
TRAIN_STEPS = 1000

def serving_input_fn():
    feature_placeholders = {
      'title': tf.placeholder(tf.string, [None]),
    }
    features = {
      key: tf.expand_dims(tensor, -1)
      for key, tensor in feature_placeholders.items()
    }
    return tflearn.utils.input_fn_utils.InputFnOps(
      features,
      None,
      feature_placeholders)

def cnn_model(features, target, mode):
    embeds = tf.contrib.layers.embed_sequence(features, vocab_size=22, embed_dim=EMBEDDING_SIZE)
    n_classes = len(labels)
    logits = tf.contrib.layers.fully_connected(embeds, n_classes, activation_fn=None)

    predictions_dict = {
        'source': tf.gather(labels, tf.argmax(logits, 1)),
        'class': tf.argmax(logits, 1),
        'prob': tf.nn.softmax(logits)
        }
    learning_rate = 0.001
    loss = tf.losses.sparse_softmax_cross_entropy(targetX, logits)
    train_op = tf.contrib.layers.optimize_loss(loss,tf.contrib.framework.get_global_step(),optimizer='Adam',learning_rate=learning_rate)

    return tflearn.ModelFnOps(mode=mode,predictions=predictions_dict,loss=loss,train_op=train_op)







def experiment_fn(output_dir):
    PADWORD='[PAD]'
    MAX_DOCUMENT_LENGTH = 3


    titles = ['Biodegradable Bags Cause Outrage in Italy','Tom Brady denies key points of ESPN Patriots article','Aldi to open first Kingwood store',PADWORD]
    labels = ['International','Sport','Business']

    TARGETS = tf.constant(["International", "Sport", "Business"])


    words = tf.sparse_tensor_to_dense(tf.string_split(titles), default_value=PADWORD)

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH)
    vocab_processor.fit(titles)

    outfilename = "/Users/eliapalme/Newsriver/Newsriver-classifier/training/vocabfile.vcb"

    vocab_processor.save(outfilename)

    nwords = len(vocab_processor.vocabulary_)


    ## Transform the documents using the vocabulary.
    XX = np.array(list(vocab_processor.fit_transform(titles)));


    # make targets numeric
    table = tf.contrib.lookup.index_table_from_tensor(mapping=TARGETS, num_oov_buckets=1, default_value=-1)
    features = tf.constant(["International", "Sport", "Business"])
    targetX = table.lookup(features)


    return tflearn.Experiment(
            tflearn.Estimator(model_fn=cnn_model, model_dir=output_dir),
            train_input_fn=XX,
            eval_input_fn=targetX,
            eval_metrics={
                'acc': tflearn.MetricSpec(
                    metric_fn=metrics.streaming_accuracy, prediction_key='class'
                )
            },
            export_strategies=[saved_model_export_utils.make_export_strategy(
                serving_input_fn,
                default_output_alternative_key=None,
                exports_to_keep=1
            )],
            train_steps = TRAIN_STEPS
        )

output_dir = "/Users/eliapalme/Newsriver/Newsriver-classifier/training/output"

learn_runner.run(experiment_fn, output_dir)

#run the graph
with tf.Session() as sess:
    #print (titles)
    tf.tables_initializer().run()
    print (targetX.eval())
    #print (sess.run(words))
    #print(x)
