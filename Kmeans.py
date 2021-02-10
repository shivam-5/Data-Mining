# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 12:08:46 2020

@author: shivam
"""

import numpy as np


class Kmeans:
    def __init__(self):
        pass

    A = None
    labelledA = None
    clusterCenters = None

    # Label the regions using defined distance metrics
    # A: matrix to be clustered
    # k: number of centers
    # centers: intial centers
    # Returns labelled version of A with center numbers and cluster centers
    def labelRegions(self, A, k, centers=None):
        self.A = A

        # Geenrate random cluster centers if None specified
        if centers is None:
            centers = np.zeros((k, A.shape[1]))
            for ii in range(0, A.shape[1]):
                low = np.min(A[:, ii]) - 1
                high = np.max(A[:, ii]) + 1
                centers[:, ii] = np.random.uniform(low, high, k)

        self.labelledA = np.zeros(A.shape[0])

        while True:
            for i in range(0, A.shape[0]):
                leastDist = float('inf')
                center = 0
                for ii in range(0, k):

                    # Choose any distance metric
                    # dist = self.sad(A[i, :] - centers[ii])
                    # dist = self.ssd(A[i, :], centers[ii])
                    dist = self.euclidean(A[i, :], centers[ii])
                    if dist < leastDist:
                        leastDist = dist
                        center = ii
                self.labelledA[i] = center

            labelCount = np.zeros(np.size(centers, axis=0))
            newCenters = np.zeros_like(centers)

            # Adjust new cluster centers and increase per label count after each iteration over whole matrix
            for i in range(0, A.shape[0]):
                point = int(self.labelledA[i])
                newCenters[point] = newCenters[point] + A[i, :]
                labelCount[point] = labelCount[point] + 1

            for i in range(0, k):
                if labelCount[i] == 0:
                    newCenters[i] = centers[i]
                else:
                    newCenters[i] = newCenters[i] / labelCount[i]

            # If new centers are not different from previous ones, we are done!
            if np.array_equal(centers, newCenters):
                break
            else:
                centers = newCenters
        self.clusterCenters = newCenters
        return self.labelledA, newCenters

    def ssd(self, a, b):
        return np.sum((a - b) ** 2)

    def sad(self, a, b):
        return np.sum(abs(a - b))

    def euclidean(self, a, b):
        return np.sqrt(np.sum((a - b) ** 2))

    # WC distance that just sums up all distances
    def withinClusterDistance(self):
        m = self.A.shape[0]
        sum = np.zeros((self.clusterCenters.shape[0], 2))
        for i in range(0, m):
            label = int(self.labelledA[i])
            sum[label][0] += self.ssd(self.A[i, :], self.clusterCenters[label])
            sum[label][1] += 1
        return np.sum(sum, axis=0)[0]

    # WC distance normalized
    def withinClusterDistance_norm(self):
        m = self.A.shape[0]
        sum = np.zeros((self.clusterCenters.shape[0], 2))
        for i in range(0, m):
            label = int(self.labelledA[i])
            sum[label][0] += np.sum((self.A[i, :] - self.clusterCenters[label]) ** 2)
            sum[label][1] += 1
        norm_sum = sum[:, 0] / sum[:, 1]
        wc = np.sum(norm_sum) / len(self.clusterCenters)
        return wc

    # WC distance that just sums up all distances
    def betweenClusterDistance(self):
        m = len(self.clusterCenters)
        sum = 0.0
        for i in range(0, m - 1):
            for ii in range(i + 1, m):
                sum += self.ssd(self.clusterCenters[i], self.clusterCenters[ii])
        return sum
