#!/usr/bin/env python
import threading, logging, time
import json
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from kafka import KafkaConsumer, KafkaProducer
from tensorflow.contrib import learn

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

class Producer(threading.Thread):
    daemon = True

    def run(self):
        producer = KafkaProducer(bootstrap_servers=['kafka.marathon.cluster.newsriver.io:9092'])



class Consumer(threading.Thread):
    daemon = True

    def run(self):
        consumer = KafkaConsumer('raw-article',
                         group_id='tf-classifier',
                         bootstrap_servers=['kafka.marathon.cluster.newsriver.io:9092'])

        # Evaluation
        # ==================================================
        checkpoint_file = tf.train.latest_checkpoint("./model/checkpoints")
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=FLAGS.allow_soft_placement,
              log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                # Normalised score tensor
                nscores = graph.get_operation_by_name("output/nscores").outputs[0]

                # Map data into vocabulary
                vocab_processor = learn.preprocessing.VocabularyProcessor.restore("./model/vocab")

                with open("./model/labelsMap", 'r') as fp:
                    labelsMap = json.load(fp)

                for message in consumer:

                    article = json.loads(message.value.decode("utf-8"));

                    if article['language']!='en':
                        continue

                    text =  article['title']+" "+ article['text'];

                    x_raw = [text[:min(1000,len(text))]]


                    # Collect the predictions here
                    all_predictions = []

                    x_test = np.array(list(vocab_processor.transform(x_raw)))

                    #all_predictions = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
                    batch_predictions,batch_nscores = sess.run([predictions,nscores], {input_x: x_test, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])

                    batch_nscores[batch_nscores < 0.2] = 0
                    batch_categories = []
                    for _score in batch_nscores:
                        categories = []
                        for i in range(len(_score)):
                            if _score[i]>0:
                                categories.append([labelsMap[str(i)],_score[i]])
                        batch_categories.append(categories)

                    print ("%s \t%s \t%s\n\n" % (batch_categories,article['title'], article['url']))



def main():
    threads = [
        Producer(),
        Consumer()
    ]

    for t in threads:
        t.start()

    while True:
        time.sleep(120)

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s.%(msecs)s:%(name)s:%(thread)d:%(levelname)s:%(process)d:%(message)s',
        level=logging.INFO
        )
    main()
