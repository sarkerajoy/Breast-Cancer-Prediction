# import pandas & train_test_spilt
import pandas as pd
from sklearn.model_selection import train_test_split

# read csv file
data=pd.read_csv('data.csv')

#drop id and target columns
data=data.drop(columns=['id','Unnamed: 32'])


#spilt data into input & output
x=data.iloc[:,1:]
y=data['diagnosis']

# spilt data into train & test data 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=5)

# import model
from sklearn.neighbors import KNeighborsClassifier

# initialize model and fit the data 
model=KNeighborsClassifier(n_neighbors=11)
model.fit(x_train,y_train)

# predict test data 
y_predict=model.predict(x_test)

# import accuracy score
from sklearn.metrics import accuracy_score

# count the accuracy 
accuracy_score(y_predict,y_test)

# collect accuracy data for different neighbors
estimator=[]
Accuracy_score=[]
def score(model,x_train,x_test,y_train,y_test):
    for i in range(1,51):
        model=KNeighborsClassifier(n_neighbors=i,algorithm='auto', leaf_size=30, 
             metric='minkowski', metric_params=None, 
             n_jobs=None, p=2, 
             weights='uniform')
        model.fit(x_train,y_train)
        y_predict=model.predict(x_test)
        acc_score=accuracy_score(y_predict,y_test)
        estimator.append(i)
        Accuracy_score.append(acc_score)
        
score(model,x_train,x_test,y_train,y_test)

# import matplotlib for visualization
import matplotlib.pyplot as plt

#plot accuracy Score vs neighbor data
plt.plot(estimator,Accuracy_score)
plt.xlabel('n_neighbor')
plt.ylabel('AccuracyScore')

# import cross val for cross validation
from sklearn.model_selection import cross_val_score

# cross validation
cross_score=cross_val_score(model,x,y,cv=15)

#cross validaion for different neighbors
cross_score=[]
n=[]
for i in range(1,51):
    n.append(i)
    knn=KNeighborsClassifier(n_neighbors=i)
    cv_score=cross_val_score(knn,x,y,cv=15)
    cross_score.append(cv_score.mean())

# misclassifiation error
MCE=[1-x for x in cross_score]

# plot misclassification error vs neighbor plot
plt.figure(figsize = (10, 6))
plt.plot(n,MCE)
plt.xlabel('n_neighbors')
plt.ylabel('Misclassification_error')
plt.show()



