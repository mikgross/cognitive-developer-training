# Cognitive Developer Training

## 27-05-2019
* (Kaggle platform)[https://www.kaggle.com] is a great source of datasets, competitions and discussions. It is good to use for learning purposes in Datascience.
* working with SQl and noSQL databases are pretty much the same
* (scikit tutorials and more)[https://scikit-learn.org]

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
    * Accuracy: Tp+Tn/(n)
    * F1 harmonic mean: 2*((P*R)/(P+R))
* usually on a training set of 1 you would have
    * Training: 0.7
    * Testing: 0.3

### Data Processing

### Hands on exercises:

**Exercise 1**
* [hands-on-1.py](hands-on/hands-on-1.py)

**Exercise 2**
* writing with PyPDF2 library text into other objects
* use the PdfFileReader and PdfFileWriter objects

**Exercise 3**
* bag of words exercise
* we get some text files and we try to arrange them in a bag of words format
* bag of words will help us to understand texts by infering meaning
* [to exercise](hands-on/task%202/Bag_of_Words.py)

----

## 28-05-2019

### TF-IDF
* look at slide for accurate description
* TF: 3/100 = 0.03
* IDF: log(10000000/1000) = 4
* TF * IDF measure: says how important te document is important to the document and how relevant it is to the collection
* use library scikit learn to vectorize tf idf
* scikit learn is the largest library for Data Science: very good for learning data science

### Clustering
* Unspuervised machine learning
* General artifical intelligence on machine learning: regardless of the data processed
* different clustering methods are suitable for specific tasks
* On data processing we are done when w represent text as vectors
* Check chosing the right algorythm
* Check excel file for exercise (manual clustering)[hands-on/task%203/exercise/clustering_script.py]

### Regression
* 70% training, 30% testing
* sum of squares needs to be the lower possible
* General training: Test set and training set
* Second one: k-fold cross validation: split your set into 5-6
* sample size should be around 
* see slide 107 for types of regressions
* 

### overfitting
* ways to deal with:
    * decrease the number of sample
    * she will send some docs

----
## 29-05-2019

### panda library

Import like this: (goes usually with numpy)
```python
import panda as pd
import numpy as np
```

### Classification
* KNN: select the class based on nearest neighgbours
    * always seect an odd number of K
* 