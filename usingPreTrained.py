import pickle

import gensim.downloader as api
import matplotlib
import nltk.data
import pymysql
from pandas import get_dummies

import ownEmbedding

matplotlib.use('agg')
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import warnings
import matplotlib.pyplot as plt
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.layers import GRU
import SlangTranslation

warnings.filterwarnings('ignore')
plt.style.use('ggplot')
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()

from keras.models import Sequential
from keras.layers import Dense


if __name__ == '__main__':
    data = pd.read_csv(
        'C:/Users/Tudor/PycharmProjects/tweeterSentiment/Misc/corpus/processedTrainingSet.csv',
        encoding='latin-1')
    data = data.drop(data.columns[0], axis=1)

    print(data)

    # data = data.sample(frac=1)
    # data = data[:10000]

    # data = data[~data.iloc[:, 1].str.startswith('@')]

    data_class_0 = data[data.iloc[:, 0] == 0]  # Negative
    data_class_1 = data[data.iloc[:, 0] == 4]

    data_class_0_under = data_class_0.sample(data_class_1.shape[0])
    data_test_under = pd.concat([data_class_0_under, data_class_1], axis=0)
    ##should be 500K tweets left at this point by using undersampling and removing the retweets

    print(data_test_under.shape)

    data = data_test_under

    data[data.columns[1]] = data[data.columns[1]].astype(str)

    externalTesting = pd.read_csv(
        'C:/Users/Tudor/PycharmProjects/tweeterSentiment/Misc/corpus/processedExternalValidationSet.csv',
        encoding='latin-1')

    externalTesting = externalTesting.drop(externalTesting.columns[0], axis=1)

    print(externalTesting)

    num_words = 1000

    tokenizer = Tokenizer(num_words=num_words, split=' ')
    tokenizer.fit_on_texts(data[data.columns[1]])
    # At this point, after reading again the preprocessed set, python seems to believe that a certain
    # row is a float type. I assume That due to preprocessing, the tweet became just an empty sentence
    # that python is interpreting as a null, since numbers have previously been removed
    X = tokenizer.texts_to_sequences(data[data.columns[1]].values)
    X = pad_sequences(X, maxlen=40)
    Y = pd.get_dummies(data[data.columns[0]]).values

    X_EXT_TEST = tokenizer.texts_to_sequences(externalTesting[externalTesting.columns[1]].values)
    X_EXT_TEST = pad_sequences(X_EXT_TEST, maxlen=40)

    Y_EXT_TEST = get_dummies(externalTesting[externalTesting.columns[0]]).values

    print(X.shape)

    print(Y.shape)

    X_TRAIN, X_TEMP, Y_TRAIN, Y_TEMP = train_test_split(X, Y,
                                                        stratify=Y,
                                                        test_size=0.5,
                                                        random_state=42)

    X_TEST, X_VAL, Y_TEST, Y_VAL = train_test_split(X_TEMP, Y_TEMP,
                                                    stratify=Y_TEMP,
                                                    test_size=0.5,
                                                    random_state=42)

    epochs = 3
    batch_size = 512

    word2vec_model = api.load("glove-wiki-gigaword-50")

    model = Sequential()
    emb = word2vec_model.wv.get_keras_embedding(train_embeddings=False)
    model.add(emb)
    model.add(GRU(128, dropout=0.2, return_sequences=True))
    model.add(GRU(128, dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])


    history = model.fit(X_TRAIN, Y_TRAIN, epochs=epochs, batch_size=batch_size, verbose=2,
              validation_data=(X_TEST, Y_TEST))


    """START PLOTTING""
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']
    xc=range(5)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('number of epochs')
    plt.ylabel('loss')
    plt.title('train vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    print (plt.style.available)
    plt.style.use(['classic'])

    plt.figure(2, figsize=(7, 5))
    plt.plot(xc, train_acc)
    plt.plot(xc, val_acc)
    plt.xlabel('number of epochs')
    plt.ylabel('accuracy')
    plt.title('train acc vs val_acc')
    plt.grid(True)
    plt.legend(['train_acc', 'val_acc'])
    print(plt.style.available)
    plt.style.use(['classic'])
    """""""""""""""""""""""""""
    """END PLOTTING"""

    filename = '25-05-2020-TESTING-PRETRAINEDWORDS-3EPOCHS-PREPROCESSING-MORNING.sav'
    pickle.dump(model, open(filename, 'wb'))
    model = pickle.load(open(filename, 'rb'))

    """START TESTING"""
    (loss, accuracy) = model.evaluate(X_TEST, Y_TEST, batch_size = batch_size, verbose = 1)
    print('[INFO] loss={:.4f}, accuracy: {:.4f}%'.format(loss, accuracy * 100))
    print("*" * 100)
    #################################################################

    (loss, accuracy) = model.evaluate(X_EXT_TEST, Y_EXT_TEST, batch_size=batch_size, verbose=1)
    print('[INFO] loss={:.4f}, accuracy: {:.4f}%'.format(loss, accuracy * 100))
    print("*" * 100)
    #################################################################

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
##################END TESTING################################


