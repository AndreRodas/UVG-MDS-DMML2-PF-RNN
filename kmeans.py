# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 18:49:20 2020

@author: jmanc
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

blobs,labels = make_blobs(n_samples=1000,centers=4,n_features=2,shuffle=True,random_state=31)
plt.scatter(blobs[:,0],blobs[:,1])
plt.show()

def calculate_WSS(points,kmax):
    sse = []
    for k in range(1,kmax+1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0
        
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i,0] - curr_center[0]) ** 2 + (points[i,1]-curr_center[1])**2
        
        sse.append(curr_sse)
    return sse


def calculate_Sil(points,kmax):
    sil = []
    kmax = 10
    for k in range(2, kmax+1):
        kmeans = KMeans(n_clusters=k).fit(points)
        labels = kmeans.labels_
        sil.append(silhouette_score(points,labels,metric='euclidean'))
    return sil

sses = calculate_WSS(blobs,10)
plt.plot(range(1,11),sses)
plt.show()

sils = calculate_Sil(blobs,10)
plt.plot(range(2,11),sils)
plt.show()

optimal_k = np.argmax(sils)+2

print("Optimal K: " + str(optimal_k))

kmeans = KMeans(n_clusters=optimal_k).fit(blobs)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

plt.scatter(blobs[:,0],blobs[:,1],c=labels)
plt.scatter(centroids[:,0],centroids[:,1],c='red')
plt.show()

kmeans.predict([[-2.5,5]])

