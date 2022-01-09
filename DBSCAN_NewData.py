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
from sklearn.neighbors import NearestNeighbors
import pandas as pd


path = './data/'
filename = "w2.txt"
databrut = pd.read_csv(path+filename, sep=" ", encoding = "ISO-8859-1", skipinitialspace=True)
data = databrut
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

plt.figure()
plt.scatter(f0_scaled, f1_scaled, s=8)
plt.title("Donnees standardisées")

########################################################################
# Run DBSCAN clustering method 
# for a given number of parameters eps and min_samples
#  
distance=0.2
min_pts=3
start = time.time()
cl_pred = cluster.DBSCAN(eps=distance, min_samples=min_pts).fit_predict(data_scaled)
end = time.time()

plt.figure()
# Plot results
plt.scatter(f0_scaled, f1_scaled, c=cl_pred, s=8)
# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(cl_pred)) - (1 if -1 in cl_pred else 0)
plt.title("Clustering DBSCAN  for " + filename +" using e="+str(distance)+" and n_neighbors="+str(min_pts)+" and Nb of clusters="+str(n_clusters_))
print('Estimated number of clusters: %d' % n_clusters_)


fig = plt.figure()
neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(data_scaled)
distances, indices = nbrs.kneighbors(data_scaled)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.plot(distances)



plt.show()