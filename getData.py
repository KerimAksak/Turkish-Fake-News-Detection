import os
import tweepy as tw
import pandas as pd

consumer_key= 'consumer_key'
consumer_secret= 'consumer_secret'
access_token= 'access_token'
access_token_secret= 'access_token_secret'

auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

# Define the search term and the date_since date as variables
search_words = "#aşı"
date_since = "2020-11-01"

# Collect tweets
tweets = tw.Cursor(api.search,
              q=search_words,
              lang="tr",
              since=date_since).items(300000)

f = open("dataset.txt", "a")

for tweet in tweets:
    print(tweet.text)
    f.write(tweet.text)

f.close()