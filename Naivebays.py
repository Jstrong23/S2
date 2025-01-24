import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

#load the dataset
iris = load_iris()
X = iris.data  
Y = iris.target


#split data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

class AI:
    def fit (self, X, Y):
        self.classes = np.unique(Y)
        self.mean = {}
        self.priors = {}
        self.varience = {}

        for i in (self.classes):
            self.mean[i] = np.mean(X)
            self.priors[i] = (X[Y == i].shape[0]) / X.shape[0]
            self.varience[i] = np.var(X)

    #---------------------------------------------------
    def Gaussian_prob(self, X, mean, varience):
        numerator = np.exp(-((X - mean) ** 2) / (2 * varience))
        denominator = np.sqrt(2 * np.pi * varience)
        return numerator / denominator
    #this gausian_probability function was written by Chat GPT (openAI, 2025), it calculates the probability of continuous data sitting at points on a curve known as the normal distribution curve.
    
    #use the gausian prob function to calculate the probability of each plant to be of each species
    def probability(self, X):
        prob = {}
        for i in (self.classes):
            prior = self.priors[i]
            prob[i] = prior + np.sum(np.log(self.Gaussian_prob(X, self.mean[i], self.varience[i])))
        return prob

    #use the probabilites to predict which secies the plant is
    def predict(self, X):
        Y_pred = []
        for i in X:
            probabilities = self.probability(X)
            Y_pred.append(max(probabilities, key=probabilities.get))
        return np.array(Y_pred)
        
#Train naive bays
nb = AI()
nb.fit(X_train, Y_train)

#Predict on the test set
Y_pred = nb.predict(X_test)

#calculate the accuracy
accuracy = np.mean(Y_pred == Y_test)
print(f"Accuracy: {accuracy * 100}")