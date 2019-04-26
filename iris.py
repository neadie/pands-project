# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 20:42:06 2019

@author: SineadF
"""


import exploratoryDataAnalysis as eda

def main():
    iris = eda.exploratoryDataAnalysisClass()
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
    
   
        




if __name__ == "__main__":
    main()