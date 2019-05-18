##############################################################################
#
#    Mahnoor Anjum
#    manomaq@gmail.com
#    References:
#        SuperDataScience,
#        Official Documentation
#
#
##############################################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# iloc gets data via numerical indexes
# .values converts from python dataframe to numpy object
dataset = pd.read_csv('Clusters.csv')
X = dataset.iloc[:, 1:3].values
y = dataset.iloc[:, 3].values

plt.scatter(X[:,0], X[:,1])
plt.show()

'''
Hierarchical Clustering

The agglomerative hierarchical clustering algorithms available in this 
program module build a cluster hierarchy that is commonly displayed as a 
tree diagram called a dendrogram. They begin with each object in a separate
cluster. At each step, the two clusters that are most similar are joined 
into a single new cluster. Once fused, objects are never separated. 
The eight methods that are available represent eight methods of defining the
similarity between clusters.

1- Single Linkage
Also known as nearest neighbor clustering, this is one of the oldest and most famous of the hierarchical
techniques. The distance between two groups is defined as the distance between their two closest members. It
often yields clusters in which individuals are added sequentially to a single group. 

2- Complete Linkage
Also known as furthest neighbor or maximum method, this method defines the distance between two groups as
the distance between their two farthest-apart members. This method usually yields clusters that are well separated
and compact.

3- Simple Average
Also called the weighted pair-group method, this algorithm defines the distance between groups as the average
distance between each of the members, weighted so that the two groups have an equal influence on the final
result.

4- Centroid
Also referred to as the unweighted pair-group centroid method, this method defines the distance between two
groups as the distance between their centroids (center of gravity or vector average). The method should only be
used with Euclidean distances.

5- Backward links may occur with this method. These are recognizable when the dendrogram no longer exhibits its
simple tree-like structure in which each fusion results in a new cluster that is at a higher distance level (moves
from right to left). With backward links, fusions can take place that result in clusters at a lower distance level
(move from left to right). The dendrogram is difficult to interpret in this case.

6- Median
Also called the weighted pair-group centroid method, this defines the distance between two groups as the
weighted distance between their centroids, the weight being proportional to the number of individuals in each
group. Backward links (see discussion under Centroid) may occur with this method. The method should only be
used with Euclidean distances.

7- Group Average
Also called the unweighted pair-group method, this is perhaps the most widely used of all the hierarchical cluster
techniques. The distance between two groups is defined as the average distance between each of their members.

8- Ward’s Minimum Variance
With this method, groups are formed so that the pooled within-group sum of squares is minimized. That is, at
each step, the two clusters are fused which result in the least increase in the pooled within-group sum of squares

Flexible Strategy
Lance and Williams (1967) suggested that a continuum could be made between single and complete linkage. The
program lets you try various settings of these parameters which do not conform to the constraints suggested by
Lance and Williams.

'''

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
plt.clf()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 10, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 10, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 10, c = 'green', label = 'Cluster 3')
plt.title('Clusters')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.show()