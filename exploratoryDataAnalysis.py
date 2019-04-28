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
import matplotlib.pyplot as plt

import seaborn as sns

class exploratoryDataAnalysisClass:
    url = ''
    attributes = []
    
    def loadDataSet(self):
        data = urllib.request.urlopen(self.url)
        df = pd.read_csv(data, sep=',')
        df.columns = self.attributes
        return df
    
    def getBoxPlot(self,kind,image):
        # self.loadDataSet().plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
        self.loadDataSet().boxplot(by=kind, figsize=(10, 10))
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
        
    def violinPlot(self,kind,var1,var2,var3,var4,image):
        plt.figure(figsize=(15,10))
        plt.subplot(2,2,1)
        sns.violinplot(x='class',y='petal_length',data=self.loadDataSet())
        plt.subplot(2,2,2)
        sns.violinplot(x='class',y='petal_width',data=self.loadDataSet())
        plt.subplot(2,2,3)
        sns.violinplot(x='class',y='sepal_length',data=self.loadDataSet())
        plt.subplot(2,2,4)
        sns.violinplot(x='class',y='sepal_width',data=self.loadDataSet())
        plt.savefig(image) 
        plt.show()
        
    def corrmatrix(self,image):
        plt.matshow(self.loadDataSet().corr())
        plt.xticks(range(len(self.loadDataSet().columns)), self.loadDataSet().columns)
        plt.yticks(range(len(self.loadDataSet().columns)), self.loadDataSet().columns)
        plt.colorbar()
        plt.savefig(image)
        plt.show() 
      
  


