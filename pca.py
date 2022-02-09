import itertools
import numpy as np
import matplotlib as mlp
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from sklearn import cluster
from sklearn import datasets
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from time import time

##Pre-process everything and set up PCA model

faces = pd.read_csv('faces.csv', header = None)
n_samples, n_features = faces.shape

faces_centered = faces - faces.mean(axis=0)
faces_centered -= faces_centered.mean(axis=1).values.reshape(n_samples, -1)

n_features = H*W
mean_image = faces_data.mean(axis=0) 
faces_data_centered = faces_data - faces_data.mean(axis=0) 
faces_data_centered -= faces_data_centered.mean(axis=1).values.reshape(n_samples, -1)

def plot_gallery(title, images, n_col=3, n_row=2, cmap=plt.cm.gray):
    plt.figure(figsize=(2.0 * n_col, 2.26 * n_row))
    plt.suptitle(title, size=16)
    for i, comp in enumerate(images):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(
            comp.reshape(image_shape),
            cmap=cmap,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        plt.xticks(())
        plt.yticks(())
    plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.0)
    

pca = decomposition.PCA()
pca.fit(faces_centered)


##Define graphing method and plot faces in selected range
plt.figure(figsize=(16, 16));
for ii in range(16):
    plt.subplot(4, 4, ii + 1) # It starts with one
    plt.imshow(pca.components_[ii].reshape(64, 64), cmap=plt.cm.gray)
    plt.grid(False);
    plt.xticks([]);
    plt.yticks([]);


##Investigate "explained variance ratio over component"
with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(10, 10));
    plt.title('Explained Variance Ratio over Component');
    plt.plot(pca.explained_variance_ratio_);
with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(16, 12));
    plt.title('Cumulative Explained Variance over EigenFace');
    plt.plot(pca.explained_variance_ratio_.cumsum());


##Actual PCA applied on faces data and result graphed
n_row, n_col = 1, 15
n_components = n_row * n_col
imsize = (64, 64)
faces= pd.read_csv('faces.csv', header = None)
n_samples, n_features = faces.shape

# global centering
faces_centered = faces - faces.mean(axis=0)

# local centering
faces_centered -= faces_centered.mean(axis=1).values.reshape(n_samples, -1)
estimator = decomposition.PCA(n_components=n_components, svd_solver="randomized", whiten=True)
data = faces_centered
estimator.fit(data)
for i, comp in enumerate(estimator.components_[:n_components]):
        plt.subplot(n_row, n_col, i + 1)
        vmax = max(comp.max(), -comp.min())
        plt.imshow(
            comp.reshape(imsize),
            cmap=plt.cm.gray,
            interpolation="nearest",
            vmin=-vmax,
            vmax=vmax,
        )
        plt.xticks(())
        plt.yticks(())
plt.subplots_adjust(0.01, 0.05, 0.99, 0.93, 0.04, 0.0)
plt.show()

##PCA variance ratio explained, investigate "explained variance ratio over component"
with plt.style.context('fivethirtyeight'):
    plt.figure(figsize=(10, 10));
    plt.title('Explained Variance Ratio over Component');
    plt.plot(estimator.explained_variance_ratio_);


##Projection by taking the first n number of principal components as a recognition technique
x0 = faces[:1]
loadings = pd.DataFrame(estimator.components_)
proj=[]
for i in range(1, 15):
    n_row, n_col = 1, i
    n_components = n_row * n_col
    estimator = decomposition.PCA(n_components=n_components, svd_solver="randomized", whiten=True)
    data = faces_centered
    estimator.fit(data)
    loadings = pd.DataFrame(estimator.components_.T)
    P = np.dot(loadings, loadings.T)
    proj.append(np.matmul(x0, P))
for i in range(1, len(proj)):
    plt.figure(figsize=(16, 16));
    p = proj[i].values.reshape(imsize)
    plt.figure
    plt.subplot(1, len(proj), len(proj))
    plt.imshow(p, interpolation='nearest')
    plt.xticks()
    plt.yticks()
    plt.show()