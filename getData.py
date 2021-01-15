#python libraries
import datetime
import tweepy
import csv
#Keys and access for Twitter
consumer_key = 'consumer_key'
consumer_secret = 'consumer_secret'
access_token = 'access_token'
access_token_secret = 'access_token_secret'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

filename = 'twitter_csvData'+'.csv' 

date_since = "tw_tarihi"
user_name = "tw_kullanıcı_ismi"
search_words = "aranacak_kelime"

tweets_data = []
with open (filename, 'a+', newline='') as csvFile:
    csvWriter = csv.writer(csvFile)
    for tweet in tweepy.Cursor(api.user_timeline, id=user_name, q = search_words, lang = "tr", since=all, count =1).items(1): 
        tweets_encoded = tweet.text.encode('utf-8')
        tweets_decoded = tweets_encoded.decode('utf-8')
        #parsed_tweet = api.get_status(tweet.id, tweet_mode='extended').full_text
        csvWriter.writerow([date_since, tweet.user.screen_name, tweet.user.name, tweet.text, tweet.retweet_count, tweet.favorite_count])
        #Direkt olarak veriler csv formatından kaydediliyor.
