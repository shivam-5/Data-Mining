import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np


# ----------------------------------------------------------------------------------------------------------- #
# Utility APIs
# ----------------------------------------------------------------------------------------------------------- #


# Returns the range of attribute values over all instances in a defined string format
def attribute_range(attribute):
    return "[{0},{1}]".format(np.min(attribute), np.max(attribute))


# Return the between cluster distance for all the cluster centers
def between_cluster_distance(cluster_centers):
    m = len(cluster_centers)
    bc = 0.0
    for i in range(0, m - 1):
        for ii in range(i + 1, m):
            bc += np.sum((cluster_centers[i] - cluster_centers[ii]) ** 2)
    return bc

# ----------------------------------------------------------------------------------------------------------- #
# Reading and refining data
# ----------------------------------------------------------------------------------------------------------- #

# Read data file to a dataframe
whole_cust = pd.read_csv('./data/wholesale_customers.csv')

# Drop 'Channel' and 'Region' columns
whole_cust = whole_cust.drop(columns=['Channel', 'Region'])

# ----------------------------------------------------------------------------------------------------------- #
# Q1 This constitutes Question 1 of Clustering
# ----------------------------------------------------------------------------------------------------------- #

output = pd.DataFrame(index=whole_cust.columns)

# Retrieve mean value for each attribute over all instances
output['mean'] = whole_cust.mean()

# Retrieve range of values for each attribute with custom api @attribute_range
output['range'] = whole_cust.apply(attribute_range, axis=0)
print(output)


# ----------------------------------------------------------------------------------------------------------- #
# Q2 This constitutes Question 2 of Clustering
# ----------------------------------------------------------------------------------------------------------- #

# Create a KMeans classifier from scikit-learn
# and fit it with the customer data
# using k = 3
km = KMeans(n_clusters=3)
km.fit(whole_cust)

# Create 15 scatter plots between each pair of attributes
plt.figure()
plt.subplots_adjust(wspace=0.3, hspace=0.7)
pltIdx = 1
for i in range(0, whole_cust.columns.size):
    for ii in range(i + 1, whole_cust.columns.size):
        plt.subplot(5, 3, pltIdx)

        # Scatter both columns representing each cluster with different color
        plt.scatter(whole_cust.iloc[:, i], whole_cust.iloc[:, ii], c=km.labels_)
        plt.xlabel(whole_cust.columns[i])
        plt.ylabel(whole_cust.columns[ii])
        pltIdx += 1
# plt.show()


# ----------------------------------------------------------------------------------------------------------- #
# Q2 This constitutes Question 2 of Clustering
# ----------------------------------------------------------------------------------------------------------- #

k_range = [3, 5, 10]
output = pd.DataFrame(columns=['k = 3', 'k = 5', 'k = 10'], index=['BC', 'WC', 'BC/WC'])

# Create a KMeans classifier from scikit-learn
# and fit it with the customer data
# using range of k values
for k in k_range:
    km = KMeans(n_clusters=k)
    km.fit(whole_cust)

    # Get Within-Cluster value
    wc = km.inertia_

    # Get Between-Cluster value
    bc = between_cluster_distance(km.cluster_centers_)

    # Get BC/WC value
    score = bc / wc

    column = "k = " + str(k)
    output[column] = [wc, bc, score]
print(output)
