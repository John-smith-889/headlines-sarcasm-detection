"""
Goal:
    
    Based on attached data build a model that will classify headline 
    types. Prepare a report where you describe your way of approaching 
    the problem and the steps you took to solve it. Don’t forget 
    to assess the quality of the model you have prepared.

"""

%reset

# Set path
import os
os.getcwd()
os.chdir(r"C:\path")


# import libraries
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pandas as pd

# Loading data from json file
headlines = pd.read_json('headlines.json', lines = True)


#==================#
# Data preparation #
#==================#

# Chceking for null values 
print(headlines.isnull().any(axis = 0))
# no null values detected

# Import regular expression library
import re
# cut all except special characters and digits and paste to new column
headlines['headline2'] = headlines['headline'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
# Now in Spyder GUI one may easily compare old column with new one 

# Stemming our data
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

# Create Pandas Series data structure as an explanatory variable
features = headlines['headline2'].apply(lambda x: x.split())
# each row of variable is a list with stemmed words

# Stemming and join to sentences again 
features = features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))
type(features)

# data vectorization 
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer()
# tv = TfidfVectorizer(max_features = 4000)
features = list(features)
features = tv.fit_transform(features).toarray()
features[0]
# transform text into numbers representation

#=====================
# Explore new features

type(features)
column_01 = features[:,0]
for i in range(len(column_01)):
    print(column_01[i])
max(column_01)


#================#
# Data modelling #
#================#

#============
# Test design

features_train, features_test, labels_train, labels_test \
= train_test_split(features, headlines['is_sarcastic'], \
                   test_size = .20, random_state = 0)


#==========
# Modelling

# Create instance of linear support vector classifier from Sci-kit learn
lsvc = LinearSVC()
# training the model
lsvc.fit(features_train, labels_train)
# getting the score of train and test data


#=================#
# Model assesment #
#=================#

# Check score for train dataset 
lsvc.score(features_train, labels_train)
# accuracy
# 0.9610146487574297

# Check score for test dataset
lsvc.score(features_test, labels_test)
# accuracy
# 0.8317109696742793





