import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
import logging
import json
from nltk.cluster import KMeansClusterer, \
cosine_distance
from sklearn import metrics
from numpy.random import shuffle
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

def cluster_kmean(file):
    data = pd.read_csv(file)
    print(len(data))
    print(data['text'][1])
    EXTRA_STOP_WORDS = ["http", "https", "iphone", "apple", "iphonexr", "xr", "vs", "iphonexsmax",
                        "enter", "chance", "win", "64", "gb", "via", "youtube", "comparison", "new", "appreciated"]
    stopwords = ENGLISH_STOP_WORDS.union(EXTRA_STOP_WORDS)

    tfidf_vect = TfidfVectorizer(stop_words=stopwords,min_df=50,max_df=0.9)
    dtm = tfidf_vect.fit_transform(data['text'])
    print(dtm.shape)
    num_clusters=3
    #clusterer = KMeansClusterer(num_clusters, \
    #                            cosine_distance, \
    #                            repeats=20,
    #                            avoid_empty_clusters=True
    #                            )
    #clusters = clusterer.cluster(dtm.toarray(), assign_clusters=True)
    km = KMeans(n_clusters=2).fit(dtm)


if __name__=="__main__":
    cluster_kmean('iPhoneXR_twitter_eda.csv')
    #cluster_kmean('xsiV_-o5488Comments.csv')
    #cluster_kmean('GISADzMnU4sComments.csv')