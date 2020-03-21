# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 23:39:21 2020

@author: yogo1
"""

import pandas as pd
import numpy as np
import pickle


Data_location = 'D:/1.AI/4.Cognizant Project/Yogi Project/Dataset/student info sample.csv'


df = pd.read_csv(Data_location)

print(df.head())

df = df.fillna(0)

X = df.iloc[:,0:14]

X = X.stack().pipe(lambda s: pd.Series(pd.factorize(s.values)[0], s.index)).unstack()


Y=df.iloc[:,-1]



from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split( 
        X, Y, test_size = 1/3, random_state = 0) 



from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression(random_state = 0) 
classifier.fit(xtrain, ytrain) 


y_pred = classifier.predict(xtest) 


from sklearn.metrics import confusion_matrix 
cm = confusion_matrix(ytest, y_pred) 
  
print ("Confusion Matrix : \n", cm) 


from sklearn.metrics import accuracy_score 
print ("Accuracy : ", accuracy_score(ytest, y_pred))    


# Saving model to disk
pickle.dump(classifier, open('D:/1.AI/4.Cognizant Project/Yogi Project/Dataset/model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('D:/1.AI/4.Cognizant Project/Yogi Project/Dataset/model.pkl','rb'))
print("Model Prediction for given data is : ", model.predict([500, 2, 400, 415, 415, 4, 4, 4, 4, 4, 4, 4, 17, 400]))



