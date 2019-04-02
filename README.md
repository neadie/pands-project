# Programing and Scripting Project

## Research
### Background information about the Iris Dataset
The data set consists of 150 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). ref(1)

The dataset contains five attributes - petal length, petal width, sepal length, sepal width and species.

## Preparation

There were 5 key libraries installed. Below is a list of the Python SciPy libraries required for this project:

- scipy
- numpy
- matplotlib
- pandas
- sklearn 



## Investigation
1.  Download the data set and write some Python code to investigate it.
2.  Summarise the data set by, for example, calculating the maximum, minimum and
mean of each column of the data set. A Python script will quickly do this for you.


## Summary of  Investigations.
 Include supporting tables and graphics as you deem necessary.

```python
# Load libraries
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from urllib import urlretrieve
import seaborn as sns
#  Load Dataset
iris = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

urlretrieve(iris)
df = pd.read_csv(iris, sep=',')
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df.columns = attributes
# Shape
print(df.shape)


# peak at the Data
print(df.head(20))


# Summary stattistics

print(df.describe())

print(df.groupby('class').size())

# box and whisker plots
df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

# histograms
df.hist()
plt.show()



# scatter plot matrix
scatter_matrix(df)
plt.show()
```

### Evaluate  Algorithms
To create some models of the data and estimate their accuracy on unseen data.
Separate out a validation dataset.
Set-up the test harness to use 10-fold cross validation.
Build 5 different models to predict species from flower measurements
Select the best model.

### Make Predictions
The KNN algorithm is very simple and was an accurate model based on the tests. Now we want to get an idea of the accuracy of the model on our validation set.

This will give us an independent final check on the accuracy of the best model. It is valuable to keep a validation set just in case you made a slip during training, such as overfitting to the training set or a data leak. Both will result in an overly optimistic result.

We can run the KNN model directly on the validation set and summarize the results as a final accuracy score, a confusion matrix and a classification report

## References 
1. Scikit-Learn , The Iris Dataset,https,<https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html>
2. Curran Kelleher,The Iris Dataset, https://gist.github.com/curran/a08a1080b88344b0c8a7
3. Jason Brownlee on June 10, 2016 in Python Machine Learning,Your First Machine Learning Project in Python Step-By-Step, <https://machinelearningmastery.com/machine-learning-in-python-step-by-step/>
4. Scikit-learn Classifiers on Iris Dataset <https://www.kaggle.com/chungyehwang/scikit-learn-classifiers-on-iris-dataset>
5. Ritchie Ng,Evaluating a Classification Model <https://www.ritchieng.com/machine-learning-evaluate-classification-model/>
