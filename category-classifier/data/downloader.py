#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import re
import logging
import sys
import subprocess
import csv




logger = logging.getLogger('Downloader')


from datetime import datetime
from elasticsearch import Elasticsearch

es = Elasticsearch(["elasticsearch-5.6.marathon.services.newsriver.io"], maxsize=5)

CATEGORIES = ['International','Business','Technology','Sports','Entertainment','Italy','Culture','Swiss','TV & Movies','Wine & Dine','Life','Motors','Science','Photography','Health']

language = 'it'





MAX_RESULTS = 50000

for category in CATEGORIES:

    query = {
        "query": {
            "query_string" : {
                "default_field" : "title",
                "query" : "metadata.category.category:\"{}\" AND language:\"{}\"".format(category,language)
            }
        }
    }


    response = es.search(index="newsriver*", body=query ,scroll='1m')
    print("Got {} Hits for category:{} and language:{}".format(response['hits']['total'],category,language))
    scroll = response['_scroll_id']
    scan = True
    with open('{}.{}.csv'.format(language,category), 'w',encoding='utf8') as f_t:
        writer = csv.writer(f_t,delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        count = 0;
        while len(response['hits']['hits']) and scan:

            for hit in response['hits']['hits']:
                title=hit["_source"]['title']
                date = hit["_source"]['discoverDate']
                url  = hit["_source"]['url']
                text = hit["_source"]['text']
                country = hit["_source"]['metadata']['category']['country']
                writer.writerow([date,title,url,text,country])
                count+=1
                if count >= MAX_RESULTS :
                    scan = False
                    break

            response = es.scroll(scroll_id = scroll, scroll = '1m')
            scroll = response['_scroll_id']
