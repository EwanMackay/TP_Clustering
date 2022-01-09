# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:08:23 2021

@author: huguet
"""
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn import preprocessing
import pandas as pd
import sklearn as sk

##################################################################
# READ a data set (arff format)

# Parser un fichier de données au format arff
# datanp est un tableau (numpy) d'exemples avec pour chacun la liste 
# des valeurs des features

# Note 1 : 
# dans les jeux de données considérés : 2 features (dimension 2 seulement)
# t =np.array([[1,2], [3,4], [5,6], [7,8]]) 
#
# Note 2 : 
# le jeu de données contient aussi un numéro de cluster pour chaque point
# --> IGNORER CETTE INFORMATION ....
#    2d-4c-no9.arff

fr='./data/pluie.csv'
databrut = pd.read_csv(fr,encoding="latin-1")
datanp1 = databrut.to_numpy()   #converts to array
datanp = datanp1[:,1:-1]    #drops first and last column

########################  NORMALISATION  ##########################
normalized_data = preprocessing.normalize(datanp)
normalized_data
 
# normalized_data = preprocessing.StandardScaler().fit_transform(datanp)
# normalized_data

#########################  AVEC OU SANS PCA #########################


##########################    CLUSTERING   #####################


########################   RESULTATS  ################################
# Run clustering method for a given number of clusters
knp = list(range(2,11))
scores = []
tps1 = time.time()
print("------------------------------------------------------")
print("Appel KMeans pour une valeur de k fixée (données init)")

for k in knp:
    model_km = cluster.KMeans(n_clusters=k, init='k-means++')
    model_km.fit(datanp)
    tps2 = time.time()
    labels_km = model_km.labels_
    # Nb iteration of this method
    iteration = model_km.n_iter_
    
    # Résultat du clustering
    # plt.scatter(f0, f1, c=labels_km, s=8)
    # plt.title("Données (init) après clustering")
    # plt.show()
    print("nb clusters =",k,", nb iter =",iteration, ", runtime = ", round((tps2 - tps1)*1000,2),"ms")
    #print("labels", labels_km)
    # Some evaluation metrics
    # inertie = wcss : within cluster sum of squares
    inert = model_km.inertia_
    silh = metrics.silhouette_score(datanp, model_km.labels_, metric='euclidean')
    print("Inertie : ", inert)
    print("Coefficient de silhouette : ", silh)
    scores.append(silh)
    
# maxValue=0.0
# kmax=0
# for i in range knp:
#     if scores[i]>maxValue:
#         maxValue=scores[i]
#         kmax=i
        
        
plt.plot(knp, scores)
plt.show()
#print("K optimal : ", kmax)

########################################################################
# TESTER PARAMETRES METHODE ET RECUPERER autres métriques
########################################################################