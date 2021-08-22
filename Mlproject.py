# import sys
# print('python: {}'.format(sys.version))
# import scipy
# print('scipy: {}'.format(scipy.__version__))
# import numpy
# print('numpy: {}'.format(numpy.__version__))
# import matplotlib
# print('matplotlib: {}'.format(matplotlib.__version__))
# import pandas
# print('pandas: {}'.format(pandas.__version__))
# import sklearn
# print('sklearn: {}'.format(sklearn.__version__))
import pandas
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import  GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier
#loading data
url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset= read_csv(url,names=names)
#dimensions of the dataset
print(dataset.shape)
#take a peek at the data
print(dataset.head(20))
#satistical summary
print(dataset.describe())
#class distribution
print(dataset.groupby('class').size())
#univariate plots-box and whisker plots
## dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
##pyplot.show()
#histogram of the variable
##dataset.hist()
## pyplot.show()
#multivariate plots
##scatter_matrix(dataset)
## pyplot.show()

#creating a validation dataset
#splitting dataset(80%-training,20%-input)
array=dataset.values
x=array[:,0:4]
y=array[:,4]
x_train,x_validation,y_train,y_validation = train_test_split(x,y,test_size=0.2,random_state=1)

#ALGORITHM
#logistic regression
#linear discriminant analysis
#k-near neighbors
#classification and regression trees
#Gaussian naive bayes
#Support vector machine

#building models
models=[]
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

#evaluating the created models
results=[]
name=[]
for name,model in models:
    kfold = StratifiedKFold(n_splits=10,random_state=None,shuffle=True)
    cv_results=cross_val_score(model,x_train,y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s : %f (%f)' %(name,cv_results.mean(),cv_results.std()))
##from the output we knoe tht svm is goot for this dataset with 0.97 and LDA is good with 0.98

 #compare our models
# pyplot.boxplot(results,labels=names)
# pyplot.title('Algorithm Comparison')
# pyplot.show()

#make predictions on svm

model=SVC(gamma='auto')
model.fit(x_train,y_train)
predictions=model.predict(x_validation)


#evaluate our predictions
print(accuracy_score(y_validation,predictions))
print(confusion_matrix(y_validation,predictions))
print(classification_report(y_validation,predictions))
