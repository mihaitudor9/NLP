import tweepy
from sqlalchemy import false
from sqlalchemy.cprocessors import str_to_date
from tweepy import StreamListener
import pymysql
from keys.keys import *
from tweepy import Stream


class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """

    def on_status(self, status):

        #Not listeninng to the retweets in the stream, just ignoring them
        if (status.text.startswith("RT @") == False):

            ID_TWEET = status.id_str
            CREATED_AT = status.created_at
            TEXT = status.text
            LOCATION = status.user.location
            USER_CREATED_AT = status.user.created_at
            FOLLOWERS = status.user.followers_count
            LANGUAGE = status.lang


            db = pymysql.connect("localhost", "me", "@Carmen0721287743", "tweeter")
            cur = db.cursor()

            """
            cur.execute('''

                            INSERT INTO tweeter.tweets (ID_TWEET, CREATED_AT, TEXT,LOCATION,USER_CREATED_AT,FOLLOWERS,LANGUAGE)
                            VALUES
                            (ID_TWEET, CREATED_AT,TEXT,LOCATION,USER_CREATED_AT,FOLLOWERS,LANGUAGE)
                            ''')
            """

            cur.execute(
                'INSERT INTO tweets(ID_TWEET, CREATED_AT,TEXT,LOCATION,USER_CREATED_AT,FOLLOWERS,LANGUAGE) VALUES (%s,%s,%s,%s,%s,%s,%s)',
                (ID_TWEET, CREATED_AT,TEXT,LOCATION,USER_CREATED_AT,FOLLOWERS,LANGUAGE)
            )

            db.commit()
            print(cur.rowcount,'recorded inserted')

    def on_error(self, status_code):
        if status_code == 420:
            return False

if __name__ == '__main__':

    #Simulaniously downloading 2 diferent streams in ordder to compare
    db = pymysql.connect("localhost", "me", "@Carmen0721287743", "tweeter")
    #db = pymysql.connect("localhost", "me", "@Carmen0721287743", "secondTwitter")
    l = StdOutListener()

    auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth)

    cur = db.cursor()

    sql = '''CREATE TABLE IF NOT EXISTS TWEETS(
        SENTIMENT VARCHAR(10),
    	ID_TWEET VARCHAR(50),
    	CREATED_AT DATETIME,
    	TEXT VARCHAR(200),
    	LOCATION VARCHAR(100),
    	USER_CREATED_AT DATE,
    	FOLLOWERS INT,
    	LANGUAGE VARCHAR(10)
    	)'''

    cur.execute(sql)


    """
    sql = '''CREATE TABLE IF NOT EXISTS secondTweets(
        SENTIMENT VARCHAR(10),
    	ID_TWEET VARCHAR(50),
    	CREATED_AT DATETIME,
    	TEXT VARCHAR(280),
    	USER_CREATED_AT DATE,
    	FOLLOWERS INT,
    	LANGUAGE VARCHAR(10)
    	)'''
    cur.execute(sql)
    
    
    """


    stream = Stream(auth, l)
    stream.filter(track=['Bill Gates'])
    #stream.filter(track=['Jeff Bezos','Bezos'])




