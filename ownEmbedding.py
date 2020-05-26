import pymysql
from imblearn.combine import SMOTEENN
from pandas import get_dummies
import re
import warnings
import numpy as np
from imblearn.over_sampling import BorderlineSMOTE, SMOTE, ADASYN, SMOTENC, RandomOverSampler
import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import pymysql
from imblearn.over_sampling import SMOTE
from keras.layers import Embedding, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from pandas import get_dummies
from sklearn.model_selection import train_test_split
import SlangTranslation
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
import pickle
import nltk
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
from keras.models import Sequential
from keras.layers import Dense

def clean_tweet(text):
    text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', text, flags=re.MULTILINE)
    text = re.sub('@[^\s]+', '', text)
    text = re.sub('[^a-zA-z0-9\s]','',text)
    return text

def lemmatize_text(text):
    return (' '.join([lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]))


if __name__ == '__main__':

    ## derived dataset from https://www.kaggle.com/kazanova/sentiment140
    data = pd.read_csv(
        'C:/Users/Tudor/PycharmProjects/tweeterSentiment/Misc/corpus/processedTrainingSet.csv',
        encoding='latin-1')
    data = data.drop(data.columns[0], axis=1)


    print(data)

    """Since the dataset is still 1M tweets big after deleting all retweets
    it is taking quite a while to build the models with all the data"""
    #data = data.sample(frac=1)
    #data = data[:100000]




    "Start Under SAMPLING"
    """
    data_class_0 = data[data.iloc[:,0] == 0]  #Negative
    data_class_1 = data[data.iloc[:,0] == 4]  #Positive labels
    
    data_class_0_under = data_class_0.sample(data_class_1.shape[0])
    data_test_under = pd.concat([data_class_0_under, data_class_1], axis=0)
    ##should be 500K tweets left at this point by using undersampling and removing the retweets
    print(data_test_under.shape)
    data = data_test_under
    "End Under SAMPLING"
    """

    #once I reload the preprocessed dataset, a couple rows become null as a consequence
    #to the proceses, and tdhen python seemed to interpret them as float variables, leading to a
    #bug that took me a while to fix. We just tell the dataframe that the Tweet column should indeed
    #be of string type.
    data[data.columns[1]] = data[data.columns[1]].astype(str)

    externalTesting = pd.read_csv(
        'C:/Users/Tudor/PycharmProjects/tweeterSentiment/Misc/corpus/processedExternalValidationSet.csv',
        encoding='latin-1')

    externalTesting = externalTesting.drop(externalTesting.columns[0], axis=1)
    print(externalTesting)


    num_words = 5000
    tokenizer = Tokenizer(num_words=num_words, split=' ')
    tokenizer.fit_on_texts(data[data.columns[1]])
    X = tokenizer.texts_to_sequences(data[data.columns[1]].values)
    X = pad_sequences(X, maxlen=40)
    Y = pd.get_dummies(data[data.columns[0]]).values

    X_EXT_TEST = tokenizer.texts_to_sequences(externalTesting[externalTesting.columns[1]].values)
    X_EXT_TEST = pad_sequences(X_EXT_TEST, maxlen=40)

    Y_EXT_TEST = get_dummies(externalTesting[externalTesting.columns[0]]).values

    print(X.shape)
    print(Y.shape)


    #50% of the data to be used in training
    X_TRAIN, X_TEMP, Y_TRAIN, Y_TEMP = train_test_split(X, Y,
                                                        test_size=0.5,
                                                        random_state=42)


    #The other half to be splitted among validation and testing sets
    X_TEST, X_VAL, Y_TEST, Y_VAL = train_test_split(X_TEMP, Y_TEMP,
                                                        test_size=0.5,
                                                        random_state=42)

    print(X_TRAIN.shape)
    print(Y_TRAIN.shape)
    print(X_TEST.shape)
    print(X_VAL.shape)

    """Start MIXING UNDER AND OVER SAMPLING"""
    #https://imbalanced-learn.readthedocs.io/en/stable/combine.html
    """SMOTE + ENN"""
    #se = SMOTEENN(random_state=42)
    #X_TRAIN, Y_TRAIN = se.fit_sample(X_TRAIN, Y_TRAIN)
    """END SMOTE + ENN"""

    embed_dim = 512
    epochs = 5


    """START UNDER SAMPLING"""
    """Rather trivial, we separate set into 2 pieces with the same label
    and then reshape the bigger one so it matches the smaller one's dimension
    then we concatanate the two subsets
    """"""
    data_class_0 = data[data.iloc[:, 0] == 0]  # Negative
    data_class_1 = data[data.iloc[:, 0] == 4]  # Positive labels

    data_class_0_over = data_class_0.sample(data_class_1.shape[0])
    data_test_over = pd.concat([data_class_0_over, data_class_1], axis=0)

    print(data_class_0.shape)
    print(data_class_1.shape)
    
    data = data_test_over
    
    """
    """END UNDER SAMPLING"""


    """""""""""""""""""""""""""
    """
    """START ACC-LOSS PLOTTING
    print(history.history.keys())
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    #summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    """
    """END PLOTTING"""

    batch_size = 512
    epochs = 5
    """START OWN EMBEDDING LAYER MODEL"""
    model = Sequential()
    model.add(Embedding(num_words, embed_dim))
    model.add(GRU(128, dropout=0.2, return_sequences=True))
    model.add(GRU(128, dropout=0.2))
    model.add(GRU(128, dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    history = model.fit(X_TRAIN, Y_TRAIN, epochs=epochs, batch_size=batch_size, verbose=1,
                        validation_data=(X_VAL, Y_VAL))
    """END OWN EMBEDDING LAYER MODEL"""


    filename = 'ownEmbeddingBest-24-05-2020-Morning.sav'

    """"""""""LOSS AND ACCURACY ON THE TEST DATA""""""""""""""""""""""""""""""""""""""
    (loss, accuracy) = model.evaluate(X_TEST, Y_TEST, batch_size = batch_size, verbose = 1)
    print('[INFO] loss={:.4f}, accuracy: {:.4f}%'.format(loss, accuracy * 100))
    print("*" * 100)
    #############################################################################
    """"""""""LOSS AND ACCURACY ON THE SEPARATED TEST DATA - less than 300 labeled tweets"
    (loss, accuracy) = model.evaluate(X_EXT_TEST, Y_EXT_TEST, batch_size=batch_size, verbose=1)
    print('[INFO] loss={:.4f}, accuracy: {:.4f}%'.format(loss, accuracy * 100))


    filename = 'FINAL-100K-mixedSamplingOwnEmbedding5epochs.sav'
    #filename = 'FINAL-100K-Full-preoprocessing-undersapling-5Epochs.sav'
    #filename = '25-05-2020-TESTING-PRETRAINEDWORDS-3EPOCHS-PREPROCESSING-MORNING.sav'

    #pickle.dump(model, open(filename, 'wb'))
    #history = model.fit(X_TRAIN, Y_TRAIN, batch_size=batch_size, verbose=1)



    model = pickle.load(open(filename, 'rb'))


    """"""""""True positive/True negatives - ON TEST DATA""""""""""""
    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for x in range(len(X_TEST)):

        result = model.predict(X_TEST[x], batch_size=1, verbose=1)[0]

        if np.argmax(result) == np.argmax(Y_TEST[x]):
            if np.argmax(Y_TEST[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1

        if np.argmax(Y_TEST[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    print("pos_acc", pos_correct / pos_cnt * 100, "%")
    print("neg_acc", neg_correct / neg_cnt * 100, "%")

    print("*" * 100)

    """"""""""""""""""TRUE POSITIVES/NEGATIVE ON THE SEPARATE TEST FILE"""""""""""""""""""""""""''
    pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
    for x in range(len(X_EXT_TEST)):

        result = model.predict(X_EXT_TEST[x].reshape(1, X_TEST.shape[1]), batch_size=1, verbose=2)[0]

        if np.argmax(result) == np.argmax(Y_EXT_TEST[x]):
            if np.argmax(Y_EXT_TEST[x]) == 0:
                neg_correct += 1
            else:
                pos_correct += 1

        if np.argmax(Y_EXT_TEST[x]) == 0:
            neg_cnt += 1
        else:
            pos_cnt += 1

    print("pos_acc", pos_correct / pos_cnt * 100, "%")
    print("neg_acc", neg_correct / neg_cnt * 100, "%")

    print("*" * 100)
    """"""""""""""""""""""""""""END TP/TN"""""""""""""""""""""""""""""""""


    db = pymysql.connect("localhost", "me", "@Carmen0721287743", "tweeter")

    downloadedTweets = pd.read_sql_query("SELECT * FROM tweets", db)
    downloadedTweets1 = pd.read_sql_query("SELECT * FROM secondTweets", db)
    

    # even thought we downloaded tweets in languages other than english
    # let's delete the ones not in english
    # keep in mind that there's no need to delete the retweets as we don't even
    # download them, just the tweets and replies
    
    downloadedTweets = downloadedTweets[downloadedTweets['LANGUAGE'] == 'en']

    # Same preprocessing procedures as we did with the training data
    downloadedTweets['TEXT'] = downloadedTweets['TEXT'].apply(
        lambda x: SlangTranslation.translator(x))

    downloadedTweets['TEXT'] = downloadedTweets['TEXT'].apply(
        lambda x: lemmatize_text(x))

    downloadedTweets['TEXT'] = downloadedTweets['TEXT'].apply(
        lambda x: clean_tweet(x))

    downloadedTweets['TEXT'] = downloadedTweets['TEXT'].str.lower()


    for row_index,row in downloadedTweets.iterrows():
        tweet = [row['TEXT']]
        text = [tweet]
        print(text)
        text = tokenizer.texts_to_sequences(text)
        text = pad_sequences(text, maxlen=40, dtype='int32', value=0)

        sentiment = (model.predict(text, batch_size=1, verbose=2, steps=None)[0])

        print(sentiment)
        print("x" * 100)

        if (sentiment[0] > 0.60):
            print ("negative")
        elif (sentiment[1] > 0.60):
            print ("positive")
        else:
            print("neutral")



    twt = tokenizer.texts_to_sequences(twt)
    #padding the tweet to have exactly the same shape as `embedding_2` input
    twt = pad_sequences(twt, maxlen=28, dtype='int32', value=0)
    print(twt)
    sentiment = model.predict(twt,batch_size=1,verbose = 2)[0]
    if(np.argmax(sentiment) == 0):
         print("negative")
    elif (np.argmax(sentiment) == 1):
         print("positive")
