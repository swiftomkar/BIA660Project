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
    text = data["text"]
    print(text[0])
    EXTRA_STOP_WORDS = ["http", "https", "iphone", "apple", "iphonexr", "xr", "vs", "iphonexsmax",
                        "enter", "chance", "win", "64", "gb", "via", "youtube", "comparison", "new","appreciated"]

    # define custom stop words
    stopwords = ENGLISH_STOP_WORDS.union(EXTRA_STOP_WORDS)

    tfidf_vect = TfidfVectorizer(stop_words=stopwords,min_df=50,max_df=0.9)

    dtm = tfidf_vect.fit_transform(text)
    print(dtm.shape)
    for nClust in range (2,7):
        num_clusters = nClust
        clusterer = KMeansClusterer(num_clusters, \
                                    cosine_distance, \
                                    repeats=20,
                                    avoid_empty_clusters=True
                                    )
        clusters = clusterer.cluster(dtm.toarray(), assign_clusters=True)

        #print(len(clusters[3421:4021]),len(labels))
        #clusters=clusters[3421:4021]
        data=pd.DataFrame()
        data["cluster"] = clusters
        centroids = np.array(clusterer.means())
        sorted_centroids = centroids.argsort()[:, ::-1]

        voc_lookup = tfidf_vect.get_feature_names()
        for i in range(num_clusters):
            # get words with top 20 tf-idf weight in the centroid
            top_words = [voc_lookup[word_index] \
                     for word_index in sorted_centroids[i, :20]]
            print("number of clusters:",nClust," Cluster %d:\n %s " % (i, "; ".join(top_words)))
        print('sklearn- kmeans')

        km=KMeans(n_clusters=nClust).fit(dtm)
        #labels = km.predict()  # labels of shape [1000,] with values 0<= i <= 9
        centroidsskl = km.cluster_centers_
        sorted_centroidsskl = centroidsskl.argsort()[:, ::-1]
        for i in range(num_clusters):
            # get words with top 20 tf-idf weight in the centroid
            top_words = [voc_lookup[word_index] \
                     for word_index in sorted_centroidsskl[i, :20]]
            print("number of clusters:",nClust,"Cluster %d:\n %s " % (i, "; ".join(top_words)))
        print('---------------------------------------------------------')



if __name__=="__main__":
    cluster_kmean('xsiV_-o5488Comments.csv')
    #cluster_kmean('GISADzMnU4sComments.csv')