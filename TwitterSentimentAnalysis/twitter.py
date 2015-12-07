from __future__ import division
import sentiment_mod as s

from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json


ckey = CKEY
csecret = CSECRET
atoken = ATOKEN
asecret = ASECRET

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)
        
        tweet = all_data["text"]
        sentiment_value, confidence = s.sentiment(tweet)

        print (tweet, sentiment_value, confidence)

        # only output twiiter feed if 4 of 5 classifiers classify the same 
        if confidence  >= 0.80:
        		output = open('twitter-out.txt', 'a')
        		output.write(sentiment_value)
        		output.write('\n')
        		output.close()
        
        return True

    def on_error(self, status):
        print status

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track = ['49ers'])