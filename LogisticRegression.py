# Importing Libraries
import numpy as np
import pandas as pd
from flask import Flask,jsonify,request
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression
import pickle
import os
from flasgger import Swagger
import flasgger
def train_and_save_model():
    '''This function creates and saves a Binary Logistic Regression
    Classifier in the current working directory
    named as LogisticRegression.pkl
    '''
    ## Creating Dummy Data for Classificaton from sklearn.make_classification
    ## n_samples = number of rows/number of samples
    ## n_features = number of total features
    ## n_classes = number of classes - two in case of binary classifier
    
    ### Import Datset
    df = pd.read_csv("b_data.csv")
    # we change the class values (at the column number 2) from B to 0 and from M to 1
    df.iloc[:,1].replace('B', 0,inplace=True)
    df.iloc[:,1].replace('M', 1,inplace=True)
    
    ### Splitting Data
    
    X = df[['texture_mean','area_mean','concavity_mean','area_se','concavity_se','fractal_dimension_se','smoothness_worst','concavity_worst', 'symmetry_worst','fractal_dimension_worst']]
    y = df['diagnosis']
    
    ## Train Test Split for evaluation of data - 20% stratified test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=0)

    #### Data Preprocessing
    
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    
    x_train = scaler.fit_transform(X_train)
    x_test = scaler.transform(X_test)

     ## Building Model
    clf_lr = LogisticRegression()
    ## Training the Model
    clf_lr.fit(x_train, y_train)
    ## Getting Predictions
    predictions = clf_lr.predict(x_test)
    
    
    ## Analyzing valuation Metrics
    print("Accuracy Score of Model : "+str(accuracy_score(y_test,predictions)))
    print("Classification Report : ")
    print(str(classification_report(y_test,predictions)))
    ## Saving Model in pickle format
    ## Exports a pickle file named Logisitc Regrssion in current working directory
    output_path = os.getcwd()
    file_name = '/LogisticRegression.pkl'
    output  = open(output_path+file_name,'wb')
    pickle.dump(clf_lr,output)
    output.close()

train_and_save_model()