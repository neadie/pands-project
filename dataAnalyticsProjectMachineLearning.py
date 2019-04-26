# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 23:32:10 2019

@author: SineadF
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:42:06 2019

@author: SineadF
"""
import urllib.request
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns

class MahcineLeanringClass:
    url = ''
    attributes = []
    
    def loadDataSet(self):
        data = urllib.request.urlopen(self.url)
        df = pd.read_csv(data, sep=',')
        df.columns = self.attributes
        return df
    
    def getBoxPlot(self,image):
        self.loadDataSet().plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
        plt.savefig(image) 
        plt.show()
        
    def getHistrogram(self,image):
        self.loadDataSet().hist()
        plt.savefig(image) 
        plt.show()
        
    def getScatterMatrix(self,image):
        sns.pairplot(data=self.loadDataSet(),hue="class",palette="Set2")
        plt.savefig(image) 
        plt.show()
      
    @staticmethod    
    def checkAlgorithmModels(seed,X_train,Y_train,scoring):
        models = []
        models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC(gamma='auto')))
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)
            cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(msg)
        return models,results,names
    
    @staticmethod
    def compareAlgorithms(names,results):
         fig = plt.figure()
         fig.suptitle('Algorithm Comparison')
         ax = fig.add_subplot(111)
         plt.boxplot(results)
         ax.set_xticklabels(names)
         plt.show()
         
    @staticmethod
    def getknnClassifierPredictions(X_train, X_validation, Y_train, Y_validation):
        # Make predictions on validation dataset
        knn = KNeighborsClassifier()
        # Fit the classifier to the data
        knn.fit(X_train, Y_train)
        # Predict the labels of the training data
        predictions = knn.predict(X_validation)
        print("Prediction: {}".format(predictions))
        print(accuracy_score(Y_validation, predictions))
        print(confusion_matrix(Y_validation, predictions))
        print(classification_report(Y_validation, predictions))




