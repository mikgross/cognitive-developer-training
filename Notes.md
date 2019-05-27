# Cognitive Developer Training

## 27-05-2019

* https://www.kaggle.com is a great source of datasets, competitions and discussions. It is good to use for learning purposes in Datascience.
* working with SQl and noSQL databases are pretty much the same
### What is machine learning

Type of machine learning depending on the dataset:
* Supervised
    * pre-labeling the data, classification is done by the human before
    * can help in churn prediction
* Semi-supervised
    * reinforcement learning; trying to feed data to the system while the system is working
    * Feeding pacman while it's playing about the state and reward --> Die or not
* Unsupervised 
    * Do not provide information on the dataset
    * no hints to what groups do the data belongs
    * up to the data scientist to decide what the dataset should be labelled

Most popular languages in datascience:
* Python
    * Definitely suitable for NLP thanks to libraries and for production purposes
    * Most popular language in ML
* R
* C++
* Sas
* Matlab

### Introduction to python

Anaconda: has lots of packages and let's you work in different environments. Install it from the web anaconda.com

### CRISP-DM
* Standard way to work in Data Mining
* Developed in the 90s and used as of today. website link: http://crisp-dm.eu

### NLP
* Words are called tokens
* steps to NLP:
    * 1) remove punctuation
    * 2) tokenization: putting separate words in array/list
    * 3) lowercasing: very important to recognize all similar words
    * 4) stopwords: Libraries removing them automatically the most common words
    * 5) Stemming: removing suffixes and prefixes
    * 6) Lemmatization: bring the word to its base meaning
    * 7) Bag-of-words: transformation of words into numbers
* Tokenization is done differently accross languages and idioms
* few important problems:
    * personnal names, companies, geo locations -> determining what named entities are and if they are not
    * very important when it comes to document processing
    * chatbot and anything into production needs it to understand intention correctly
* Some calculations to compute the quality of your model
    * precision: TP/(TP + FP)
    * recall: TP/(FN+TP)
    * F1 harmonic mean: 2*((P*R)/(P+R))
* usually on a training set of 1 you would have
    * Training: 0.7
    * Testing: 0.3

### Data Processing

### Hands on exercises:

**Exercise 1**
* print row
* get input value from keyboard



----

## 28-05-2019

----
## 29-05-2019