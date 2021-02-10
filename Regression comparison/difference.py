# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:40:15 2020

@author: shivam
"""

import math

class Difference:
    
    #  Calculate sum of absolute distance
    def SAD(self, X, Y):
        if (len(X) != len(Y)):
            print("Size not same!")
            return
        absDiff = 0
        for i in range(0,len(X)):
            absDiff = absDiff + abs(X[i] - Y[i])
        return absDiff

    #  Calculate sum of squared distance
    def SSD(self, X, Y):
        if (len(X) != len(Y)):
            print("Size not same!")
            return
        diffSquare = 0
        for i in range(0,len(X)):
            diffSquare = diffSquare + (X[i] - Y[i]) * (X[i] - Y[i]);
        return diffSquare

    #  Calculate Euclidean distance
    def Eclidean(self, X, Y):
        if (len(X) != len(Y)):
            print("Size not same!")
            return
        return  math.sqrt(self.SSD(X,Y));
    
    def dp(self,P1,P2):
        len1 = len(P1)
        len2 = len(P2)
        if len1 != len2:
            print("size not same")
            return
        sum = 0
        for i in range(0,len1):
            sum = sum + (P1[i] - P2[i])**2
        return math.sqrt(sum)