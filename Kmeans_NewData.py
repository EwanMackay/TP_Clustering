# -*- coding: utf-8 -*-
"""
@author: Ewan
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import time

import pandas as pd

from sklearn.metrics import silhouette_score 

from sklearn.cluster import KMeans

from sklearn.preprocessing import StandardScaler
from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from mpl_toolkits.mplot3d import Axes3D


path = './data/'
filename = "w2.txt"

def initData(filename):
    #return arff.loadarff(open(path+filename, 'r'))
    return pd.read_csv(path+filename, sep=" ", encoding = "ISO-8859-1", skipinitialspace=True)

def standardise(raw):
    scaler = StandardScaler()
    return scaler.fit_transform(raw)
    
def plot2d(filename, doStandardise=False):
    databrut = initData(filename)
    datanp = databrut.to_numpy()
    if doStandardise:
        scaler = StandardScaler()
        datanp = scaler.fit_transform(datanp)
    text = filename + (" standardisées" if doStandardise else "")
    print("---------------------------------------")
    print("Affichage données 2d : " + text)
    f0 = datanp[:,0] # tous les élements de la première colonne
    f1 = datanp[:,1] # tous les éléments de la deuxième colonne
    plt.figure()
    plt.scatter(f0, f1, s=8)
    plt.title("Données 2D : " + text)

## CLUSTERING METHOD TO BE CALLED
def clusterKMeans(filename, nbCluster, doStandardise=False):
    start = time.time()
    databrut = initData(filename)
    datanp = databrut.to_numpy()
    if doStandardise:
        datanp = standardise(datanp)
    label = KMeans(n_clusters=nbCluster, random_state=0).fit_predict(datanp)
    end = time.time()
    text = filename + " with " + str(nbCluster) + " clusters"
    print("---------------------------------------")
    print("Affichage données 2d : " + text)
    plt.figure()
    for index in range(nbCluster):
        filteredLabel = datanp[label == index]
        color = (np.random.random(),np.random.random(),np.random.random())
        plt.scatter(filteredLabel[:,0] , filteredLabel[:,1], c = [color], s=8)
    plt.title("K-means clustering for " + text)
    
## FUNCTION TO EVALUATE USING THE SILHOUETTE
def evaluateKMeansSilhouette(filename, maxNbClusters, doStandardise=False):
    start = time.time()
    silhouette_avg = []
    K = range(2,maxNbClusters)
    databrut = initData(filename)
    datanp = databrut.to_numpy()
    plt.figure()
    if doStandardise:
        datanp = standardise(datanp)
    for num_clusters in K:
        # initialise kmeans
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(datanp)
        cluster_labels = kmeans.labels_

        # silhouette score
        silhouette_avg.append(silhouette_score(datanp, cluster_labels))
    end = time.time()
    print("Silouhette analysis - Elapsed time")
    plt.plot(K,silhouette_avg,"bx-")
    plt.xlabel("Values of K") 
    plt.ylabel("Silhouette score") 
    plt.title("Silhouette analysis For Optimal k")

## DISPLAYING FROM FUNCTIONS
plot2d(filename, True)
clusterKMeans(filename, 3, True)

plt.show()


