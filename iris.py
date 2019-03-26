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


# Split-out validation dataset
array = df.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
scoring = 'accuracy'
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)



# Spot Check Algorithms
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



# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()



# Make predictions on validation dataset
knn = KNeighborsClassifier()
# Fit the classifier to the data
knn.fit(X_train, Y_train)
# Predict the labels of the training data
predictions = knn.predict(X_train)
print("Prediction: {}".format(predictions))
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))



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
