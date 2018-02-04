from tensorflow.contrib import predictor
import tensorflow as tf
from kafka import KafkaConsumer, KafkaProducer
import threading, logging, time
import json
import glob
from dataset_service import DataSetService
from article import Article


samplesFiles = glob.glob('./outputdir/export/Servo/*')
predict_fn = predictor.from_saved_model(samplesFiles[0])


KFKA_HOSTS = ["kafka.marathon.services.newsriver.io:9092"]

FEEDS = {}


class Producer(threading.Thread):
    daemon = True

    def run(self):
        producer = KafkaProducer(bootstrap_servers=KFKA_HOSTS)



class Consumer(threading.Thread):
    daemon = True

    def run(self):
        consumer = KafkaConsumer('raw-article',
                         group_id='tf-classifier',
                         bootstrap_servers=KFKA_HOSTS)

        dataset = DataSetService()

        for message in consumer:

            article = Article()
            article.loadJSON(json.loads(message.value.decode("utf-8")))


            if article.language!='it':
                continue
            if len(article.referrals) != 1:
                continue

            referral = article.referrals[0]


            if 'category' in referral:
                category_expected = referral['category']
            else:
                continue

            if category_expected == None:
                continue

            text =  article.title+" "+ article.text;
            words = text.strip().lower().split(" ")
            predictions = predict_fn({"text": [words]})

            category_predicted = predictions['source'][0].decode("utf-8")
            prob  = predictions['prob'][0][predictions['class'][0]]

            hasError = 0
            if category_predicted!=category_expected :
                print("Class:{:.2f} => {}  \t\t Expected:{} Title:{}".format(prob,category_predicted,category_expected,article.title))
                hasError= 1

            feedStat = None
            referralURL = referral['referralURL']

            if referralURL in FEEDS:
                feedStat = FEEDS[referralURL]
                feedStat['errors']+=hasError
                feedStat['total']+=1
            else:
                feedStat ={}
                feedStat['errors']=hasError
                feedStat['total']=1
                FEEDS[referralURL]=feedStat


            if feedStat['total'] > 2:
                errorRate = feedStat['errors']/feedStat['total']
                feedStat ={}
                feedStat['errors']=0
                feedStat['total']=0
                FEEDS[referralURL]=feedStat
                #if errorRate > 0.5:
                print("Rate:{:.2f}  \t\t Feed:{}".format(errorRate,referralURL))

                #dataset.addSample(article)




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
