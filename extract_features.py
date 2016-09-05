import codecs
import json
from os import listdir
from os.path import isfile, join
import csv
import numpy
import sys
from datetime import datetime
import time
from nltk.tokenize import RegexpTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from stop_words import get_stop_words
from gensim import corpora, models
from string import digits
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import KMeans, MiniBatchKMeans


# extract features
def followee_per_follower_ratio(user_data):
    return user_data['friends_count'] * 1.0 / user_data['followers_count']

def number_of_distinct_user_to_reply(posts):
    replies_to = set([post['in_reply_to_user_id'] for post in posts if post['in_reply_to_user_id'] is not None])
    return len(replies_to)

def proportion_of_replies(posts):
    replies = sum([1 for post in posts if post['in_reply_to_status_id_str'] is not None])
    return replies * 1.0 / len(post)

def url_in_profile(user_data):
    if 'url' in user_data['entities'].keys() and 'urls' in user_data['entities']['url'].keys() and len(user_data['entities']['url']['urls']) > 0:
        return True
    if user_data['entities']['description'] is not None and len(user_data['entities']['description']['urls']) > 0:
        return True
    return False

def usernames_at_description(user_data):
    return (0, 1)[user_data['screen_name'] in user_data['description'] or user_data['name'] in user_data['description']]

def tweets_per_month(user_data):
    months = (datetime.now() - datetime.strptime(user_data['created_at'], '%a %b %d %H:%M:%S +0000 %Y')).days * 1.0 / 30
    return user_data['statuses_count'] * 1.0 / months

def user_features(user_data, posts):
    return [
        user_data['id'],
        user_data['screen_name'],
        user_data['followers_count'],
        user_data['listed_count'],
        user_data['statuses_count'],
        user_data['friends_count'], # Number of Followings (B
        user_data['favourites_count'],
        usernames_at_description(user_data),
        tweets_per_month(user_data),
        followee_per_follower_ratio(user_data),
        # user_data['created_at'],
        # URL in the profile (Boolean);
        (0, 1)[url_in_profile(user_data)],
        # 28) Description size
        len(user_data['description']),
        # Proportion of tweets that are replies;
        proportion_of_replies(posts),
        # Numberofdistinctuserstowhomtheuser replied.
        number_of_distinct_user_to_reply(posts)
    ]

def number_of_hashtag(posts):
    number_of_hashtags = [1 for post in posts if len(post['entities']['hashtags']) > 0]
    return sum(number_of_hashtags) * 1.0 / len(posts)

def number_of_urls(posts):
    number_of_urls = [1 for post in posts if len(post['entities']['urls']) > 0]
    return len(number_of_urls) * 1.0 / len(posts)

def words_size(posts):
    word_lengths = [len(word) for post in posts for word in tokenizer.tokenize(post['text'])]
    return numpy.mean(word_lengths)

def number_of_retweet(posts):
    retweet_counts = [1 for post in posts if post['retweet_count'] > 0]
    return len(retweet_counts) * 1.0 / len(posts)

def number_of_favourite(posts):
    favorite_counts = [1 for post in posts if post['favorite_count'] > 0 if 'retweeted_status' not in post.keys()]
    return len(favorite_counts) * 1.0 / len(posts)

def update_frequency(user, posts):
    created_at_times = [time.mktime(datetime.strptime(post['created_at'], '%a %b %d %H:%M:%S +0000 %Y').timetuple()) for post in posts]
    created_at_times.sort()
    diff = numpy.diff(created_at_times)
    return numpy.mean(diff)

def update_frequency_sd(user, posts):
    created_at_times = [time.mktime(datetime.strptime(post['created_at'], '%a %b %d %H:%M:%S +0000 %Y').timetuple()) for post in posts]
    created_at_times.sort()
    diff = numpy.diff(created_at_times)
    return numpy.std(diff)

def number_of_posts_retweeted(posts):
    retweet_counts = [1 for post in posts if 'retweeted_status' not in post.keys() and post['retweeted'] is True]
    return sum(retweet_counts) * 1.0 / len(posts)

def number_of_posts_retweet(posts):
    retweet_counts = [post['retweet_count'] for post in posts if 'retweeted_status' not in post.keys()]
    return sum(retweet_counts) * 1.0 / len(posts)

def number_of_posts_favourite(posts):
    favourite_counts = [post['favorite_count'] for post in posts if 'retweeted_status' not in post.keys()]
    return sum(favourite_counts) * 1.0 / len(posts)

def number_of_direct_messages(user, posts):
    direct_messages = [1 for post in posts if 'retweeted_status' not in post.keys() and ('user_mentions' in post['entities'].keys())]
    return sum(direct_messages) * 1.0 / len(posts)

def word_per_posts(user, posts):
    word_per_post = [len(post['text'].split()) for post in posts if 'retweeted_status' not in post.keys()]
    return sum(word_per_post) * 1.0 / len(posts)

def proportion_of_retweets_among_tweets(posts):
    retweet_counts = [1 for post in posts if 'retweeted_status' in post.keys()]
    return sum(retweet_counts) * 1.0 / len(posts)

# extract bag-of-words feature
def train_text_model(folder, num_clusters):
    tweets_data = read_tweets_data(folder)
    
    transformer = TfidfVectorizer().fit(tweets_data)
    X_train_tfidf = transformer.transform(tweets_data)
    
    km = KMeans(n_clusters=num_clusters, random_state=1)
    print "learning kmeans %s " % num_clusters
    km.fit(X_train_tfidf)
    
    return transformer, km

def text_features(transformer, km, tweets_data, num_clusters):
    clean_tweets_data = preprocess_tweets_data(tweets_data)
    tfidf = transformer.transform(clean_tweets_data)
    features = km.predict(tfidf)
    features = numpy.bincount(features)
    features.resize((1, num_clusters))
    return features.tolist()[0]

def posts_features(user, posts):
    return [
        number_of_hashtag(posts),
        number_of_urls(posts),
        number_of_retweet(posts),
        number_of_favourite(posts),
        update_frequency(user, posts), # in seconds
        update_frequency_sd(user, posts), # in seconds
        proportion_of_retweets_among_tweets(posts),
        number_of_posts_retweeted(posts),
        word_per_posts(user, posts),
        number_of_posts_retweet(posts),
        number_of_posts_favourite(posts)
    ]
        

# preprocess tweets data
def preprocess_tweets_data(tweets_data):
    # clean url links
    return [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet, flags=re.MULTILINE) for tweet in tweets_data]

# read and write data
def read_json_file(folder):
    filepaths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and f.endswith('.json')]
    
    tweets_data = []
    
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            for line in f:
                tweet = json.loads(line)
                tweets_data.append(tweet['text'])
    return tweets_data

def preprocess_tweet(tweets_data):
    
    # clean url links
    tweets = [re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', tweet, flags=re.MULTILINE) for tweet in tweets_data]
    
    # tokenize tweet data
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = [tokenizer.tokenize(tweet.lower()) for tweet in tweets]
    
    # remove stop words from token
    en_stop = get_stop_words('en')
    uni_grams = [i for token in tokens for i in token if len(i)==1]
    bi_grams = [i for token in tokens for i in token if len(i)==2]
    stoplist  = set(en_stop + uni_grams + bi_grams)
    tokens = [[i for i in token if i not in stoplist] for token in tokens]
    
    # remove numbers
    tokens = [[i for i in token if len(i.strip(digits)) == len(i)] for token in tokens]
    
    return tokens

def read_tweets_data(folder):
    filepaths = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f)) and f.endswith('.json')]
    
    tweets_data = []
    for filepath in filepaths:
        with open(filepath, 'r') as f:
            for line in f:
                tweet = json.loads(line)
                tweets_data.append(tweet['text'])
    
    return tweets_data

def to_csv(folder, csv_filename, transformer=None, km=None, num_clusters=None):
    filepaths = [f for f in listdir(folder) if isfile(join(folder, f)) and f.endswith('.json')]
    
    with open(csv_filename, 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        
        for filepath in filepaths:
            with open(join(folder, filepath), "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    user = json.loads(lines[0])['user']
                    posts = [json.loads(line) for line in lines]
                    tweets = [post['text'] for post in posts]
                    influencer = (1, 0)['non_influencers' in filepath]
                    if km is not None:
                        text_feature = text_features(transformer, km, tweets, num_clusters)
                    else:
                        text_feature = []
                    csv_writer.writerow(user_features(user, posts) + posts_features(user, posts) + text_feature + [influencer])

                    
def read_csv(filename):
    data = numpy.genfromtxt(filename, delimiter=',')
    return data[:, 2:-1], data[:, -1]


