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
    #for nClusters in range(4,5):
        nClusters=3
        data = pd.read_csv(file)
        print(len(data))
        print(data['text'][1])
        print(len(data['Labels']))
    #data['text'] = data['text'].str.replace(r'[\\u]', '')
    #data['text'] = data['text'].str.replace(r'\n', '')
        EXTRA_STOP_WORDS = ["http", "https", "iphone", "apple", "iphonexr", "xr", "vs", "iphonexsmax",
                        "enter", "chance", "win", "64", "gb", "via", "youtube", "comparison", "new", "appreciated"]
        stopwords = ENGLISH_STOP_WORDS.union(EXTRA_STOP_WORDS)

        tfidf_vect = TfidfVectorizer(stop_words=stopwords,min_df=5)
        dtm = tfidf_vect.fit_transform(data['text'])
        print(dtm.shape)
        voc_lookup = tfidf_vect.get_feature_names()

    #clusterer = KMeansClusterer(num_clusters, \
    #                            cosine_distance, \
    #                            repeats=20,
    #                            avoid_empty_clusters=True
    #                            )
    #clusters = clusterer.cluster(dtm.toarray(), assign_clusters=True)
        km = KMeans(n_clusters=nClusters,n_init=20).fit(dtm)
        clusters = km.labels_.tolist()
        print(km.labels_)

        centroidsskl = km.cluster_centers_
        print(centroidsskl)
        sorted_centroidsskl = centroidsskl.argsort()[:, ::-1]
        for i in range(nClusters):
        # get words with top 20 tf-idf weight in the centroid
            top_words = [voc_lookup[word_index] \
                     for word_index in sorted_centroidsskl[i, :20]]
            print("number of clusters:", nClusters, "Cluster %d:\n %s " % (i, "; ".join(top_words)))

        clusters = clusters[0:150]
        labels=data['Labels'][0:150]
        datum=pd.DataFrame()
        datum["label"]=labels
        datum["clusters"]=clusters
        print(pd.crosstab(index=datum['clusters'], columns=datum['label']))

        cluster_dict = {0: 1,
                        1: 2,
                        2: 3
                        }
        target = [cluster_dict[i] for i in clusters]
        print(type(target),type(labels))
        labelList=[]
        mylables=labels.tolist()
        for i in mylables:
            labelList.append(int(i))

        print(len(labelList))
        print(len(target))
        print(type(target[0]))

        # Assign true class to cluster
        print(metrics.classification_report(labelList, target))
        plt.plot(km.labels_,data['created_at'])
        plt.show()
        print('---------------------------------------------------------')

if __name__=="__main__":
    #cluster_kmean('iPhoneXR_twitter_eda.csv')
    #cluster_kmean('iPhoneXR_twitter_eda.csv')
    cluster_kmean('iPhoneXS_twitter_eda.csv')

    #cluster_kmean('GISADzMnU4sComments.csv')