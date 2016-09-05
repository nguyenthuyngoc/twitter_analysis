from __future__ import print_function
import logging
import numpy as np
import sys

from optparse import OptionParser
from configparser import ConfigParser
import time
from pymongo import MongoClient
from twython import Twython, TwythonRateLimitError
from datetime import datetime
import csv
import traceback
import json

from os.path import isfile, join

def wait_for_awhile(twitter):
    reset = int(twitter.get_lastfunction_header('x-rate-limit-reset'))
    wait = max(reset - time.time(), 0) + 10
    print("Rate limit exceeded waiting: %sm %0.0fs" % (int(int( wait)/60),wait % 60 ))
    time.sleep(wait)

def save_user_tweets(twitter, twitter_id, prefix, folder):
	print("query data for %s" % twitter_id)
	file = open(join(folder, '%s_%s.json' % (prefix, twitter_id)), 'w')

	count = 1000
	max_id = None
	for i in range(0, (count + 1) / 200):
		params = {'screen_name': twitter_id, 'count': 200, 'contributor_details': 'true', 'max_id': max_id }
		timeline = twitter.get_user_timeline(**params)

		for index, tweet in enumerate(timeline):
			file.write(json.dumps(tweet))
			file.write("\n")

		max_id = timeline[-1]['id'] - 1

	file.close()


consumer_key = '81Q1vv1Idwir1Ta6wDMeHUsmx'
consumer_secret = 'oOFRJXJdy49Qd16txT4e0ZFuJzttiN10CS5W3MCpQgHYcGD4ra'
access_token = '2890458710-oM98r2FPtI1ozs5vhhVyOCiA18Mj6xBRUZzH7ui'
access_secret = 'C8AI48J0JqqtPT73jMm27RCMWRZG2OBNCGx68d3GyPfeh'

twitter = Twython(consumer_key, consumer_secret, access_token, access_secret)

def download_twitter_users(filename, folder):
	csv_file = open(filename, 'r')
 	csv_reader = csv.reader(csv_file)

 	data = []
	for row in csv_reader: 	
		try:
			twitter_id = row[0].lower()
			if row[2] == 'opinion_maker':
				prefix = 'influencers'
			else:
				prefix = 'non_influencers'
			save_user_tweets(twitter, twitter_id, prefix, folder)

		except TwythonRateLimitError as e:
			wait_for_awhile(twitter)
		except:
			print(" FAILED:", id)	
			print("Unexpected error:", sys.exc_info()[0])
			print('-' * 60)
			traceback.print_exc(file=sys.stdout)
			print('-' * 60)

	csv_file.close()
	return data


# 1. 
download_twitter_users("train.csv", "./data1/train")
download_twitter_users("test.csv", "./data1/test")
download_twitter_users("validation.csv", "./data1/validation")

# 2

# 3