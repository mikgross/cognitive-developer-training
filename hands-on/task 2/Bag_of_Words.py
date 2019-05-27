# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:38:23 2019

@author: Mikael Gross
"""
#Your task is to create Frequency Bag of Words

import os
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer

# init tokenizer object
tknzr = TweetTokenizer()
porter = PorterStemmer()

# init stopwards
# Reading a document
directory_path ="C:\\Users\\mikgross\\Desktop\\gitRepos\\cognitive-developer-deloitte\\hands-on\\task 2\\limited_scope"

# array for all words
all_words = []

# bag-of-words array
bag_of_words = []

# array with texts
array_with_texts = []

# Names of entries
directorie = sorted(os.listdir(directory_path))

for item in directorie:
    # Getting file path
    filePath = directory_path + '\\' + item 
    # Open and read content of text document
    file = open(filePath, 'r')
    fileText = file.read()
    file.close()
    
    # remove punctuation using 'translation table' and predefined punctuation characters
    fileText = fileText.translate({ord(i): None for i in string.punctuation})
    
    # converting to lower case
    fileText = fileText.lower()
    
    # splitting into tokens
    tokenized = tknzr.tokenize(fileText)
    
    # remove remaining tokens that are not alphabetic
    tokenized = [token for token in tokenized if token.isalpha()]
    
    # Stop words filtering
    # filter out stop words using predefined dictionary for english stopwords
    stop_words = set(stopwords.words('english'))
    tokenized = [token for token in tokenized if not token in stop_words]
    
    # stemming of words
    tokenized = [porter.stem(token) for token in tokenized]
    
    # all words from the dictionary
    all_words.extend(tokenized)
    
    # all texts from the corpus
    array_with_texts.append(tokenized)

# creating a frequency dictionary from stemmed tokens
dictionary = nltk.FreqDist(all_words)

# filling up bag-of-words array
# loop over all documents
for document in array_with_texts:
    bag_of_words_rows = []
    #dictionary for specific document
    docDic = nltk.FreqDist(document)
    #cheking if word from dictionary is in a document
    for item in dictionary:
        # if yes, add its frequency
        if item in docDic:
            bag_of_words_rows.append(docDic[item])
        # if not, write zero value
        if item not in docDic:
            bag_of_words_rows.append(0)
    bag_of_words.append(bag_of_words_rows)
