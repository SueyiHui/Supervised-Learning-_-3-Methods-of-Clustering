# [Shuyi, Hui]
# [20198085]
# [MMA]
# [2021W]
# [MMA869]
# [07/07/2020]





import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 

from sklearn.metrics import silhouette_score, silhouette_samples
import sklearn.metrics
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture


import itertools

import scipy

############################################ Answer to Question [1], Part [a]############################################################
# Read in data from Uncle Steve's GitHub repository_ JewelryData 
df = pd.read_csv("https://raw.githubusercontent.com/stepthom/sandbox/master/data/jewelry_customers.csv")

df.info()
df.head()

X = df.to_numpy()
X.shape

#Standardize Data
scaler = StandardScaler()
X = scaler.fit_transform(X)


############################################# Answer to Question [1], Part [b]##########################################################
## Model Development1_K-Means Algorithm
k_means = KMeans(init='k-means++', n_clusters=5, n_init=10, random_state=42)
k_means.fit(X)

k_means.labels_
#print(k_means.labels_)


#Internal Validation Metrics
## WCSS
silhouette_score(X, k_means.labels_)
k_means.inertia_

#Elbow Method
inertias = {}
silhouettes = {}
for k in range(2, 11):
    kmeans = KMeans(init='k-means++', n_init=10, n_clusters=k, max_iter=1000, random_state=42).fit(X)
    inertias[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
    silhouettes[k] = silhouette_score(X, kmeans.labels_, metric='euclidean')
    

plt.figure();
plt.grid(True);
plt.plot(list(inertias.keys()), list(inertias.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Inertia");
plt.savefig('out/mall-kmeans-elbow-interia.png');


plt.figure();
plt.grid(True);
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
plt.title('K-Means, Elbow Method')
plt.xlabel("Number of clusters, K");
plt.ylabel("Silhouette");
plt.savefig('out/mall-kmeans-elbow-silhouette.png');

#### According to both plots, the best number of K is 5


##Model Development2_DBSCAN Algorithmn
#HyperParameter Tuning, in this case the best fit is K = 5
db = DBSCAN(eps=0.5, min_samples=3)
db.fit(X)

db.labels_
silhouette_score(X, db.labels_)

#Elbow Method For DBSCAN
silhouettes = {}

epss = np.arange(0.1, 0.9, 0.1)
minss = [3, 4, 5, 6, 7, 8, 9, 10]

ss = np.zeros((len(epss), len(minss)))

for i, eps in enumerate(epss):
    for j, mins in enumerate(minss):
        db = DBSCAN(eps=eps, min_samples=mins).fit(X)
        if len(set(db.labels_)) == 1:
            ss[i, j] = -1
        else:
            ss[i, j] = silhouette_score(X, db.labels_, metric='euclidean')
    

plt.figure();
plt.plot(list(silhouettes.keys()), list(silhouettes.values()));
for i in range(len(minss)):
    plt.plot(epss, ss[:, i], label="MinPts = {}".format(minss[i]));
plt.plot(epss, ss[:, 1]);
plt.title('DBSCAN, Elbow Method')
plt.xlabel("Eps");
plt.ylabel("Silhouette");
plt.legend();
#### According to the graph,eps is 0.3 and onwards seems to perform a better S Score


###Dive Deep_ Parameter Exploration
def do_dbscan(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    unique_labels = set(db.labels_)
    n_clusters = len(unique_labels)
    
    if n_clusters <= 1:
        print('eps={}, min_samples={}, n_clusters <= 1. Returning.'.format(eps, min_samples))
        return
    
    sil = silhouette_score(X, db.labels_)
    print("eps={}, min_samples={}, n_clusters={}, sil={}".format(eps, min_samples, n_clusters, sil))
    

epss = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
min_samples = range(1, 10)

for prod in list(itertools.product(epss, min_samples)):
    do_dbscan(X, prod[0], prod[1])

#### According to the sil result of each eps and minpts, the best parameter is eps = 0.5, minpts = 3 and k =5


##Model Development3_Hierarchical(Agglomerative) Algorithm
agg = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
agg.fit(X)

agg.labels_
silhouette_score(X, agg.labels_)


##Curious about whether "euclidean" is the best fit for discance metrics 
def Compare_metric(X, linkage, metric):
    aggl = scipy.cluster.hierarchy.linkage(X, method=linkage, metric=metric)
    
    labels = scipy.cluster.hierarchy.fcluster(aggl, 5, criterion="maxclust")
    
    sil = 0
    n = len(set(labels))
    if n > 1:
        sil = silhouette_score(X , labels, metric=metric)
    print("Linkage={}, Metric={}, Clusters={}, Silhouette={:.3}".format(linkage, metric, n, sil))
    
    
linkages = ['complete', 'ward', 'single', 'centroid', 'average']
metrics = ['euclidean', 'minkowski', 'cityblock', 'cosine', 'correlation', 'chebyshev', 'canberra', 'mahalanobis']

for prod in list(itertools.product(linkages, metrics)):
    
    # Some combos are not allowed
    if (prod[0] in ['ward', 'centroid']) and prod[1] != 'euclidean':
        continue
        
    Compare_metric(X, prod[0], prod[1])

## According to the result, linkage = complete & metric = Correlation gives highest S score
    
      
## In order to testify,conducting method for Each Cluster, Show Visual Feature Stats
K = 5
n_features = df.shape[1]

#linkages = ['complete', 'ward', 'single', 'centroid', 'average']
#metrics = ['euclidean', 'minkowski', 'cityblock', 'cosine', 'correlation', 'chebyshev', 'canberra', 'mahalanobis']
aggl = scipy.cluster.hierarchy.linkage(X, method='complete', metric='correlation')
labels = scipy.cluster.hierarchy.fcluster(aggl, K, criterion="maxclust")
f, axes = plt.subplots(K, n_features, figsize=(24, 12), sharex='col', sharey='col')

cat_col_names = list(df.select_dtypes(include=np.object).columns)
num_col_names = list(df.select_dtypes(include=np.number).columns)
X_num = df[num_col_names].to_numpy()
X_cat = df[cat_col_names].to_numpy()

cols = [x for x in num_col_names]
cols += [x for x in cat_col_names]
rows = ['Cluster {}'.format(i+1) for i in np.arange(K)]


for i, label in enumerate(set(labels)):
    n = df.iloc[labels==label].shape[0]
  

    col_idx = 0
    for col in num_col_names:
        sns.distplot(df[[col]], hist=False, rug=False, label="All", ax=axes[i-1, col_idx]);
        chart = sns.distplot(df.iloc[labels==label][[col]], hist=False, rug=False, label="Cluster {}".format(label), ax=axes[i-1,col_idx]);
        chart.set_yticklabels([])
        col_idx=col_idx+1
        
    for col in cat_col_names:
        all_prop_df = (df[col].value_counts(normalize=True).reset_index())
        all_prop_df['Cluster']= 'All'

        prop_df = (df.iloc[labels==label][col]
           .value_counts(normalize=True)
           .reset_index())

        prop_df['Cluster']= 'Cluster {}'.format(label)

        prop_df = pd.concat([all_prop_df, prop_df])
        prop_df = prop_df.reset_index(drop=True)

        chart = sns.barplot(x='index', y=col, hue='Cluster', data=prop_df, ci=None, ax=axes[i-1, col_idx])
        #chart.set_xticklabels(chart.get_xticklabels(), rotation=55)
        
        col_idx=col_idx+1

for ax1 in axes:
    for ax in ax1:
        ax.get_legend().remove()
        ax.set_ylabel('')
        ax.set_xlabel('')
        
for ax, col in zip(axes[0], cols):
    ax.set_title(col, size='large')

for ax, row in zip(axes[:,0], rows):
    ax.set_ylabel(row, rotation=90, size='large')

for j in np.arange(n_features):
    ax = axes[K-1,j]
    if j >= len(num_col_names): #Tmp hack to avoid rotating numeric lablels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    
plt.show();


################################################################# Answer to Question [1], Part [c]###################################################
#Interpreate the Results
##For Each Cluster, Show Textual Feature Stats (Method 1)
import seaborn as sns

pd.set_option("display.precision", 2)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

cat_col_names = list(df.select_dtypes(include=np.object).columns)
num_col_names = list(df.select_dtypes(include=np.number).columns)

print('All Data:')
print('Number of Instances: {}'.format(X.shape[0]))
df.describe(include=[np.number]).transpose()
#df.describe(include=[np.float64]).transpose()
for col in cat_col_names:
    df[col].value_counts()


for i, label in enumerate(set(labels)):
    n = df.iloc[labels==label].shape[0]
      
    print('\nCluster {}:'.format(label))
    print('Number of Instances: {}'.format(n))

    df.iloc[labels==label].describe(include=[np.number]).transpose()
    #df.iloc[labels==label].describe(include=[np.object]).transpose()
    for col in cat_col_names:
        df.iloc[labels==label][col].value_counts()


##For Each Cluster, Show Textual Feature Stats (Method 2)
X1 = df.copy()
def describe_clusters(X, labels):
    X2 = X.copy()
    X2['ClusterID'] = labels
    print('\nCluster sizes:')
    print(X2.groupby('ClusterID').size())
    
    
    print(X2.groupby('ClusterID').mean())
    print(X2.groupby('ClusterID').agg(lambda x: x.value_counts().index[0]))
    
    print('\nCluster stats:')
    from IPython.display import display
    display(X2.groupby('ClusterID').describe(include='all').transpose())

describe_clusters(X1, labels)

