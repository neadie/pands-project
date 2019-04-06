# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:42:06 2019

@author: SineadF
"""

import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import dataAnalyticsProjectMachineLearning as dap










def main():
    iris = dap.MahcineLeanringClass()
    iris.url='http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris.attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df = iris.loadDataSet()
    # Shape
    print(' The shape of the iris data set')
    print(df.shape)
    print(' *************************************************')
    # peak at the Data
    print(' Peak at the Data')
    print(df.head(20))
    print(' *************************************************')
    print(' Summary stattistics')
    # Summary stattistics
    print(df.describe())
    print(' *************************************************')
    print('Compute group sizes')
    print(df.groupby('class').size())
    print(' *************************************************')
    print('Box Plots')
    iris.getBoxPlot('boxplot.png')
    print(' *************************************************')
    print('Histograms')
    iris.getHistrogram('histrogram.png')
    print(' *************************************************')
    print('Scatter Matrix ')
    iris.getScatterMatrix('scatter_matrix.png')
    print(' *************************************************')
    print('Split-out validation dataset ')
    array = df.values
    X = array[:,0:4]
    Y = array[:,4]
    validation_size = 0.20
    seed = 7
    scoring = 'accuracy'
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    print(' *************************************************')
    print('Spot Check Algorithms ')
    models,results,names=iris.checkAlgorithmModels(seed,X_train,Y_train,scoring)
    print(' *************************************************')
    print('Compare Algorithms ')
    iris.compareAlgorithms(names,results)
    print(' *************************************************')
    print('Make predictions on validation dataset Fit the classifier to the data ')
    iris.getknnClassifierPredictions(X_train, X_validation, Y_train, Y_validation)
    # example of training a final classification model
    # generate 2d classification dataset
    # fit final model
    model = LogisticRegression()
    model.fit(X, Y)
    ## Exploratory data analysis
    ## I be working with the iris dataset. My goal was  
    ## be to predict species('Iris-versicolor','Iris-virginica','Iris-setosa') based on "sepal_length", "sepal_width", "petal_length", "petal_width"
    plt.figure()
    sns.countplot(x='sepal_length', hue='class', data=df, palette='RdBu')
    plt.xticks([0,1,2], ['Iris-versicolor','Iris-virginica','Iris-setosa'])
    plt.show()
    plt.figure()
    sns.countplot(x='sepal_width', hue='class', data=df, palette='RdBu')
    plt.xticks([0,1,2], ['Iris-versicolor','Iris-virginica','Iris-setosa'])
    plt.show()
    plt.figure()
    sns.countplot(x='petal_length', hue='class', data=df, palette='RdBu')
    plt.xticks([0,1,2], ['Iris-versicolor','Iris-virginica','Iris-setosa'])
    plt.show()
    
    plt.figure()
    sns.countplot(x='petal_width', hue='class', data=df, palette='RdBu')
    plt.xticks([0,1,2], ['Iris-versicolor','Iris-virginica','Iris-setosa'])
    plt.show()
        




if __name__ == "__main__":
    main()