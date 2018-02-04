import datetime
import csv

class DataSetService:
    """ Article class represents a Newsriver article """

    def __init__(self):
        """ Create a new article at the origin """

    def addSample(self,article):
        today = datetime.date.today().strftime('%Y-%m-%d')

        with open('./data/{}.{}-{}.csv'.format(article.language,article.category,today), 'a',encoding='utf8') as f_t:
            writer = csv.writer(f_t,delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
            writer.writerow([article.date,article.title,article.url,article.text,article.country])

        return
