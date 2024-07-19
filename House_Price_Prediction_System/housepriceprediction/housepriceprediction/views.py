from django.shortcuts import render
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
import os

def home(request):
    return render(request, 'home.html')

def predict(request):
    return render(request, 'predict.html')

def result(request):
    file_path = "USA_Housing.csv"
    df = pd.read_csv(file_path)

    df = df.drop('Address', axis=1)
    
    X = df.drop('Price', axis=1)
    Y = df['Price']
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    lm = LinearRegression()
    lm.fit(X_train, Y_train)
    
    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    
    pred = lm.predict(np.array([var1, var2, var3, var4, var5]).reshape(1, -1))
    pred = round(pred[0])
    
    price = "The predicted price is $"+str(pred)
    
    return render(request, 'predict.html', {"result2":price})