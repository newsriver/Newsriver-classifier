class Article:
    """ Article class represents a Newsriver article """

    def __init__(self):
        """ Create a new article at the origin """
        self.text = ""
        self.title = ""
        self.url = ""
        self.date = ""
        self.category = None
        self.language = None
        self.country = None
        self.referrals = None

    def loadJSON(self, article_json):
        self.title = article_json['title']
        self.text = article_json['text']
        self.language = article_json['language']
        self.url = article_json['url']
        self.date = article_json['discoverDate']
        self.referrals = article_json['referrals']

        if 'metadata' in article_json and 'category' in article_json['metadata']:
            category = article_json['metadata']['category']
            self.category = category['category']
            if 'country' in category:
                self.country  = category['country']
