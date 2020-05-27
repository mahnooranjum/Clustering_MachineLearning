##############################################################################
#    Mahnoor Anjum
#    References:
#        Official Documentation
##############################################################################
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
# iloc gets data via numerical indexes
# .values converts from python dataframe to numpy object
dataset = pd.read_csv('Moons.csv')
X = dataset.iloc[0:1000, 1:3].values
y = dataset.iloc[0:1000, 3].values
plt.scatter(X[:,0], X[:,1])
plt.show()

from sklearn.cluster import Birch
model = Birch()
model.fit(X)
y_pred = model.predict(X)
# Visualising the clusters
clusters = np.unique(y_pred)
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_idx = np.where(y_pred == cluster)
	plt.scatter(X[row_idx, 0], X[row_idx, 1])

plt.show()

