# -*- coding: utf-8 -*-
"""

@author: Ewan
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
import pandas as pd


##################################################################

path = './data/'
filename = "w2.txt"
# databrut = arff.loadarff(open(path+"banana.arff", 'r'))
databrut = pd.read_csv(path+filename, sep=" ", encoding = "ISO-8859-1", skipinitialspace=True)
# datanp = np.array([[x[0],x[1]] for x in databrut[0]])
datanp = databrut.to_numpy()


########################################################################
# Preprocessing: standardization of data
########################################################################

from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(datanp)

data_scaled = scaler.transform(datanp)

import scipy.cluster.hierarchy as shc

print("-------------------------------------------")
print("Affichage données standardisées            ")
f0_scaled = data_scaled[:,0] # tous les élements de la première colonne
f1_scaled = data_scaled[:,1] # tous les éléments de la deuxième colonne
#print(f0)
#print(f1)

plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Standardised data")


# Run clustering method for a given number of clusters
print("-----------------------------------------------------------")
tps3 = time.time()
k=3
linkage='average' # CHOSE HERE THE LINKAGE METHOD
model_scaled = cluster.AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage=linkage)
model_scaled.fit(data_scaled)
#cluster.fit_predict(X)

tps4 = time.time()
labels_scaled = model_scaled.labels_
fig = plt.figure()
plt.scatter(f0_scaled, f1_scaled, c=labels_scaled, s=8)
plt.title("Dataset : " + filename + " Agglomerative Clustering (k=" + str(k) + " using linkage="+linkage+")")

# Some evaluation metrics
silh = metrics.silhouette_score(data_scaled, labels_scaled, metric='euclidean')
print("Coefficient of the silhouette : ", silh)



def evaluateACSilhouette(data_scaled, maxNbClusters, linkage):
    start = time.time()
    silhouettes = []
    K = range(2,maxNbClusters)
    fig = plt.figure()
    for num_clusters in K :
        model_scaled = cluster.AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage=linkage)
        model_scaled.fit(data_scaled)
        labels_scaled = model_scaled.labels_
        silh = metrics.silhouette_score(data_scaled, labels_scaled, metric='euclidean')
        silhouettes.append(silh)
    end = time.time()
    plt.plot(K,silhouettes,"bx-")
    plt.xlabel("Values of K") 
    plt.ylabel("Silhouette score") 
    plt.title("Optimal k - Silhouette criteria")

#evaluateACSilhouette(data_scaled, 3, linkage)   ##comment here if optimal k chosen

plt.show()

