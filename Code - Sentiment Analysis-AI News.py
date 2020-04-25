# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 15:53:30 2020
Artifical Intelligence News Coverage - Sentiment Analysis
Resources Used: https://data-science-blog.com/blog/2018/11/04/sentiment-analysis-using-python/
"""
# Loading .csv data of Media Cloud's article headlines for Top 2018 U.S. Publications including "artificial intelligence" somewhere in the text of the  article.

import pandas as pd
df1 = pd.read_csv("C:/Users/joshu/OneDrive/Documents/IST-736/HW 1/AIstoriesfull.csv", encoding = "ISO-8859-1")

#setting up environment
import matplotlib.pyplot as plt
%matplotlib inline  
import nltk
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer, PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk
nltk.download('stopwords')
#setting up environment for VADER
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
nltk.downloader.download('vader_lexicon')

#checking data size and attributes

df1.shape
df1.columns

## Data Preperation
#lowercasing headlines
df1['title'] = df1['title'].astype(str)
df1['title'][1]
df1['title'] = df1['title'].apply(lambda x: " ".join(x.lower() for x in x.split()))
df1['title'][1]

#removing special characters
df1['title'] = df1['title'].str.replace('[^\w\s]','')
df1['title'][1]

#removing stopwords
stop_words = set(stopwords.words('english'))

df1['title'] = df1['title'].apply(lambda x: " ".join(x for x in x.split() if x not in stop_words))
df1['title'][1]

## Finding the sentiment score using TextBlob
# Defining a function that calculates the score for the whole dataset
 
def senti(x):
    return TextBlob(x).sentiment  
 
df1['senti_score'] = df1['title'].apply(senti)
 
#checking data
df1.senti_score.head()

#exporting data to csv for use in Tableau
df1.to_csv(r'C:/Users/joshu/OneDrive/Documents/IST-736/HW 1/AIsentiments.csv')

## Finding the sentiment score using VADER
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(x):
    score = analyser.polarity_scores(x)
    return("{:-<40} {}".format(x, str(score)))

df1['vader_score'] = df1['title'].apply(sentiment_analyzer_scores)
df1.vader_score.head()
df1.to_csv(r'C:/Users/joshu/OneDrive/Documents/IST-736/HW 1/AIsentiments2.csv')

