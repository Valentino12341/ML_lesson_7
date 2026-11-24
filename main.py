# SVM = support vector machine

from sklearn import datasets

import matplotlib.pyplot as plt
import pandas as pd

#sklearn datasets
cancer_dict = datasets.load_breast_cancer()
print(cancer_dict.keys())

data = pd.DataFrame(cancer_dict.data)
data.columns = cancer_dict.feature_names
data["isCancer"] = cancer_dict.target

print(data.info())
print(data.head())

y = data["isCancer"]

data.drop("isCancer",axis= 1)

x = data

from sklearn.model_selection import train_test_split


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50)

from sklearn import svm


# SVC = Suport vector classifier
classifier = svm.SVC(kernel="linear") 

classifier.fit(x_train,y_train)

pred = classifier.predict(x_test)


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


accuracy = accuracy_score(y_test,pred)
accuract = round(accuracy*100,2)

print("this is:",accuract,"Percent accurate")


from sklearn.metrics import confusion_matrix
import seaborn as sns

matrix = confusion_matrix(y_test,pred)
sns.heatmap(matrix,annot=True,fmt="d")
plt.title("the title")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
