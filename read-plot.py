# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 23:08:23 2021

@author: huguet
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import sklearn as sk

from scipy.io import arff
from sklearn import cluster
from sklearn import metrics
from sklearn import decomposition

from mpl_toolkits import mplot3d
from sklearn import preprocessing
from sklearn.cluster import KMeans


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
databrut=databrut.drop(columns='Ville')
datanp = databrut.to_numpy()

#print(databrut)
print(datanp)

##################################################################
# PLOT datanp (en 2D) - / scatter plot
# Extraire chaque valeur de features pour en faire une liste
# EX : 
# - pour t1=t[:,0] --> [1, 3, 5, 7]
# - pour t2=t[:,1] --> [2, 4, 6, 8]
print("---------------------------------------")
print("Affichage données initiales            ")
f0 = datanp[:,0] # tous les élements de la première colonne
#f1 = datanp[:,1] # tous les éléments de la deuxième colonne
#f2 = datanp[:,2] # tous les éléments de la troisième colonne for 3D

#Affiche les données bruts
#print(f0)
#print(f1)
#print(f2)
######################## PRE-PROCESSING : standardisation #######################
# scaler = preprocessing.StandardScaler().fit(datanp)
# data_scaled = scaler.transform(datanp)


######################## CLUSTERING #######################
#kmeans = KMeans(n_clusters=2, random_state=0).fit(datanp)



########################### AFFICHAGE #########################

#plt.scatter(f0,f1,s=4)
plt.title("Donnees initiales")
plt.show()
# plt.scatter(data_scaled,s=4)
# plt.title("Donnees pré-traités")
# plt.show()

#fig = plt.figure()
#ax = plt.axes(projection="3d") for 3D
#ax.plot_wireframe(f0, f1, f2, color='green')   Use PCA method for wireframe
#ax.set_xlabel('x')
#ax.set_ylabel('y')
#ax.set_zlabel('z') for 3D
#ax.scatter3D(f0, f1, f2, s=2, c=f1, cmap='hsv');


