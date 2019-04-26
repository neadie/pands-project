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
    
    def getBoxPlot(self,image):
        # self.loadDataSet().plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
        self.loadDataSet().boxplot(by="class", figsize=(10, 10))
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
      
  


