import csv
import re

def translator(user_string):

    user_string = user_string.split(" ")
    j = 0
    for _str in user_string:
        # File path which consists of Abbreviations.
        fileName = "C:/Users/Tudor/PycharmProjects/tweeterSentiment/Misc/Slang/slangEng.txt"
        # File Access mode [Read Mode]
        accessMode = "r"
        with open(fileName, accessMode) as myCSVfile:
            # Reading file as CSV with delimiter as "=", so that abbreviation are stored in row[0] and phrases in row[1]
            dataFromFile = csv.reader(myCSVfile, delimiter="=")
            # Removing Special Characters.
            _str = re.sub('[^a-zA-Z0-9-_.]', '', _str)
            for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if _str.upper() == row[0]:
                    # If match found replace it with its appropriate phrase in text file.
                    user_string[j] = row[1]
            myCSVfile.close()
        j = j + 1
    # Replacing commas with spaces for final output.
    result = (' '.join(user_string))

    return result


def deleteStopWordsFromFile(text):

    text = text.split(" ")
    j = 0
    for x in text:
        # File path which consists of english stopwords.
        fileName = "C:/Users/Tudor/PycharmProjects/tweeterSentiment/Misc/stop-words/english.txt"

        accessMode = "r"
        with open(fileName, accessMode) as myCSVfile:

            dataFromFile = csv.reader(myCSVfile)

            for row in dataFromFile:
                # Check if selected word matches short forms[LHS] in text file.
                if x.upper() == row:
                    # If match found replace it with its appropriate phrase in text file.
                    x = ''
            myCSVfile.close()
        j = j + 1
    # Replacing commas with spaces for final output.
    result = (' '.join(x))

    return result
