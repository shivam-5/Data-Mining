from sklearn import datasets
import numpy as np
import random

def getMostFrequentValue(X, y, n_cls):
    clsFreq = []
    featureFrequenies = []
    # Evaluate each feature
    for i in range(0, X.shape[1]):

        featureFrequency = {}
        # Evaluate all instances for that column
        for ii in range(0, X.shape[0]):
            if X[ii, i] not in featureFrequency:
                clsList = np.zeros(n_cls)
                featureFrequency[X[ii, i]] = clsList
            (featureFrequency[X[ii, i]])[y[ii]] += 1
        featureFrequenies.append(featureFrequency)
    return featureFrequenies


# Create classification rule for 2 features
def createClassificationRules(frequencies):
    classificationRulesF1 = {}
    feature1 = frequencies[0]
    feature2 = frequencies[1]
    for f1 in feature1.keys():
        classificationRulesF2 = {}
        for f2 in feature2.keys():
            clsF1, freqF1 = np.argmax(feature1.get(f1)), np.max(feature1.get(f1))
            clsF2, freqF2 = np.argmax(feature2.get(f2)), np.max(feature2.get(f2))
            if clsF1 == clsF2:
                classificationRulesF2[f2] = clsF1
            elif freqF1 == freqF2:
                classificationRulesF2[f2] = random.choice([clsF1, clsF2])
            else:
                classificationRulesF2[f2] = clsF1 if freqF1 > freqF2 else clsF2
        classificationRulesF1[f1] = classificationRulesF2
    return classificationRulesF1


def computeError(X, y, rules):
    incorrect = 0
    for i in range(0, X.shape[0]):
        ruleValue = rules.get(X[i, 0])
        for ii in range(1, X.shape[1]):
            ruleValue = ruleValue.get(X[i, ii])
        if ruleValue != y[i]:
            incorrect += 1
    return incorrect / X.shape[0]



iris = datasets.load_iris()
data = iris.data
target = iris.target
n_classes = np.size(iris.target_names)

featureFrequencies = getMostFrequentValue(data[:, 0:2], target, n_classes)
rules = createClassificationRules(featureFrequencies)
error = computeError(data[:, 0:2], target, rules)
print("Accuracy of 1R classifier(using first 2 features only): %.2f" % (1 - error))
