# -*- coding: utf-8 -*-

"""

Created on Fri May 31 09:19:13 2019

@author: mikgross

"""

import pandas as pd
import numpy as np
import xgboost as xgb

# assign the train and test dataset
trainPath = '.\\titanic\\train.csv'
testPath = '.\\titanic\\test.csv'

# read test and train data
testData = pd.read_csv(testPath)
trainData = pd.read_csv(trainPath)

# clean data
# erase clumns tickets, cabin, embarked, name for trainData
trainData = trainData.drop(columns = 'Name')
trainData = trainData.drop(columns = 'Ticket')
trainData = trainData.drop(columns = 'Cabin')
trainData = trainData.drop(columns = 'Embarked')
# replace strings by numbers
trainData = trainData.replace(to_replace = 'male', value = 0)
trainData = trainData.replace(to_replace = 'female', value = 1)

# erase clumns tickets, cabin, embarked, name for testData
testData = testData.drop(columns = 'Name')
testData = testData.drop(columns = 'Ticket')
testData = testData.drop(columns = 'Cabin')
testData = testData.drop(columns = 'Embarked')
# replace strings by numbers
testData = testData.replace(to_replace = 'male', value = 0)
testData = testData.replace(to_replace = 'female', value = 1)

# extract labels
am = 625
trainData1 = trainData[:am]
trainData2 = trainData[am:]
labelTrain1 = trainData1['Survived']
labelTrain2 = trainData2['Survived']

trainData1 = trainData1.drop(columns = 'Survived')
trainData2 = trainData2.drop(columns = 'Survived')

# assign data to xg matrix
matTrain1 = xgb.DMatrix(trainData1, label = labelTrain1)
matTrain2 = xgb.DMatrix(trainData2, label = labelTrain2)

# sepcify booster parameters
param = {'max_depth': 4, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
param['nthread'] = 1
param['eval_metric'] = 'auc'

# specify validation
evallist = [(matTrain2, 'eval'), (matTrain1, 'train')]

# Train the model
num_round = 10
model = xgb.train(param, matTrain1, num_round, evallist)

# test the model
# prepare test matrix
matTest = xgb.DMatrix(testData)
prediction = model.predict(matTest)

# clean the prediction vector
prediction = np.where(prediction < 0.5, 0, prediction)
prediction = np.where(prediction >= 0.5, 1, prediction)
prediction = prediction.astype(int)

# create the output file
# extract first column with passenger ID from test set
out = pd.DataFrame(testData['PassengerId'])
# write teh predictions in the table
out.insert(1, 'Survived', prediction)

out
# write the table in a csv
out.to_csv(path_or_buf = 'gender_submission.csv', index = False)
