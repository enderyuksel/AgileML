# trying this machine learning beginnerhttps://machinelearningmastery.com/machine-learning-in-python-step-by-step/

# Load libraries
import pandas
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

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

print("START")

# Summarize the dataset
print ("dataset shape/dimensions")
print(dataset.shape) # 150 instances, 5 attributes
print("first 20 rows")
print(dataset.head(20))
print("statistical summary")
print(dataset.describe())
print("class distribution")
print(dataset.groupby('class').size()) # each class has the same number of instances (50, or 33% of dataset)

# data visualization

# univariate plots: to better understand each attribute
# multivariate plots: to better understand the relationships between attributes

# univariate plots (plots of each individual variable)
# box and whisker plots (as the input variables are numeric)
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()  # shows us distribution of the input attributes
# histogram
dataset.hist()
plt.show() # also shows us the distibution, for instance now we can see that speal-length and sepal-width look like Gaussian distribution (we can use this assumption to pick algorithms later)

# multivariate plots (interaction between variables)
# scatterplots of all pairs
scatter_matrix(dataset)
plt.show() # 