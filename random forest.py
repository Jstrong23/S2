import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import random
import time



#create AI class
class AI:
    def decision(self, test_data, SetosaMeans, versicolourMeans):
        setosa_total = 0
        versicolour_total = 0
        for x in  range (5):
            setosa_score = 0
            versicolour_score = 0
            vector = random.randint(1,10)/10
            total_score = 0
            if (test_data.values[0] > SetosaMeans.values[0] - vector) and (test_data.values[0] < SetosaMeans.values[0] + vector):
                setosa_score += 1
            if (test_data.values[1] > SetosaMeans.values[1] - vector) and (test_data.values[1] < SetosaMeans.values[1] + vector):
                setosa_score += 1
            if (test_data.values[2] > SetosaMeans.values[2] - vector) and (test_data.values[2] < SetosaMeans.values[2] + vector):
                setosa_score += 1
            if (test_data.values[3] > SetosaMeans.values[3] - vector) and (test_data.values[3] < SetosaMeans.values[3] + vector):
                setosa_score += 1
            if (test_data.values[0] > versicolourMeans.values[0] - vector) and (test_data.values[0] < versicolourMeans.values[0] + vector):
                versicolour_score += 1
            if (test_data.values[1] > versicolourMeans.values[1] - vector) and (test_data.values[1] < versicolourMeans.values[1] + vector):
                versicolour_score += 1
            if (test_data.values[2] > versicolourMeans.values[2] - vector) and (test_data.values[2] < versicolourMeans.values[2] + vector):
                versicolour_score += 1
            if (test_data.values[3] > versicolourMeans.values[3] - vector) and (test_data.values[3] < versicolourMeans.values[3] + vector):
                versicolour_score += 1
            if setosa_score > 3:
                setosa_total += 1
            elif versicolour_score > 3:
                versicolour_total += 1
            
        #calculate probability of each plant
        setosa_prob = setosa_total/4
        versicolour_prob = versicolour_total/4

        if setosa_prob > 0.3:
            return("setosa", setosa_prob)
        elif versicolour_prob > 0.3:
            return("versicolour", versicolour_prob)
        else:
            return("undetermined", 1 - setosa_prob - versicolour_prob)
                
#create instance of AI
ai = AI()


#---------------------------------------------
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
iris = fetch_ucirepo(id=53) 
  
# data (as pandas dataframes) 
#X = iris.data.features 
#y = iris.data.targets 
  
# metadata 
#print(iris.metadata) 
  
# variable information 
#print(iris.variables) 
#---------------------------------------------This code was created to import the iris dataset (Fisher, 1936) into python by (UCI Machine Learning Repository, https://archive.ics.uci.edu/dataset/53/iris)
start = time.time()
#choose training data
data = iris.data.features
print(data)
species = iris.data.targets
df = pd.DataFrame(data)
df_Target = pd.DataFrame(species)

#chose training data
Setosa_train = df.loc[0:34]
versicolour_train = df.loc[50:74]



#calculate means of training data
SetosaMeans = Setosa_train.mean()
versicolourMeans = versicolour_train.mean()

#print(SetosaMeans)
#print(versicolourMeans)
#print(virginicaMeans)



#choose test data
setosa_test = df.loc[35:49]
versicolour_test = df.loc[75:99]
virginica_test = df.loc[135:149]

#prep performance data
FP = 0
FN = 0
TP = 0
TN = 0
true_array = []
score_array = []

for x in range (45):
    prob = 0
    test = "setosa"
    i = x + 35
    test_data = setosa_test
    if x > 14:
        test = "versicolour"
        i += 35
        test_data = versicolour_test
        if x > 29:
            test = "virginica"
            i += 35
            test_data = virginica_test
    output, prob = ai.decision(test_data.loc[i], SetosaMeans, versicolourMeans)
    if output == test:
        correct = "true"
        TP += 1
        true_array.append(1)
    elif output == "undetermined" and test == "virginica":
        correct = "true"
        TN += 1
        true_array.append(1)
    elif output == "undetermined":
        correct = "false"
        FN += 1
        true_array.append(0)
    else:
        correct = "false"
        FP += 1
        true_array.append(0)

    score_array.append(prob)
    print (x, "---", output, "---", prob, "---", correct)

end = time.time()

print (end - start)

    
    
    
#accuracy
accuracy = (TP+FN)/(TP+TN+FP+FN)
print(accuracy)

#precision
precision = (TP)/(TP+FP)
print(precision)

#recall
recall = (TP)/(TP+FN)
print (recall)

#F1 score
F1Score = 2*((precision*recall)/(precision+recall))
print (F1Score)




    

