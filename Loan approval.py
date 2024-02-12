#1.Import libraries and dataset

#to load dataframe
import pandas as pd
#used for working with arrays and mathematics funtions
import numpy as np
#to visualize data features
import matplotlib.pyplot as plt
#to see correlation between features using heatmap
import seaborn as sns 

data = pd.read_csv(r"C:\Users\izzat\Downloads\ML Project\Loan approval\LoanApprovalPrediction.csv")

#view the imported dataset
print(data.head(5))

#check data type for every column
print(type("Loan_ID"))
print(type("Gender"))
print(type("Married"))
print(type("Dependents"))
print(type("Education"))
print(type("Self_Employed"))
print(type("ApplicantIncome"))

""" A data type object refers to columns that contain 
mixed data types or non-numeric data types"""

obj = (data.dtypes == 'object') # to check whether each column has an object data types
print("Categorical variables:",len(list(obj[obj].index)))

#Loan ID completely unique and not correlated with other column
#Drop Loan_ID column
data.drop(['Loan_ID'], axis=1, inplace=True)

#Visualize using barplot
#To show which value is dominating as per our dataset
obj = (data.dtypes == 'object')
object_cols = list(obj[obj].index)
plt.figure(figsize=(18,36)) 
index = 1
  
for col in object_cols: 
  y = data[col].value_counts() 
  plt.subplot(11,4,index) 
  plt.xticks(rotation=90) 
  sns.barplot(x=list(y.index), y=y) 
  index +=1

"""As all categorical values are binary so we can use Label Encoder
for all such columns and the values will change into int datatype"""
# Import label encoder 
from sklearn import preprocessing 
    
# label_encoder object knows how  
# to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
obj = (data.dtypes == 'object') 
for col in list(obj[obj].index): 
  data[col] = label_encoder.fit_transform(data[col])
  
# To find the number of columns with  
# datatype==object 
obj = (data.dtypes == 'object') 
print("Categorical variables:",len(list(obj[obj].index)))

#heatmap plotting
plt.figure(figsize=(12,6)) #12,6 is the size of figure 12 units width, 6 units length
sns.heatmap(data.corr(),cmap='BrBG',fmt='.2f', 
            linewidths=2,annot=True)
#data.corr()-calculate correlation matrix of the dataframe
#cmap=BrBG - colourmap used which is Brown-Blue-Green
#fmt='.2f - specifies format of annotation, 2 decimal places

sns.catplot(x="Gender", y="Married", 
            hue="Loan_Status",  
            kind="bar",  
            data=data)

for col in data.columns: 
  data[col] = data[col].fillna(data[col].mean())  
    
print(data.isna().sum())

#splitting dataset
from sklearn.model_selection import train_test_split 

"""X contains independent variable for the model, created
by dropping loan status, axis=1 specifies that the operation is performed
on the column"""  
X = data.drop(['Loan_Status'],axis=1) 

"""Y contains target variable, extracted from the column Loan_status"""
Y = data['Loan_Status']

print(X.shape,Y.shape) 
  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                                                    test_size=0.4, 
                                                    random_state=1) 
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

print('\n')
#Model training and evaluation
#To predict the accuracy we will use the accuracy score function from scikit learn library
#Accuracy predictions in training set
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import LogisticRegression 
  
from sklearn import metrics 
  
knn = KNeighborsClassifier(n_neighbors=3) 
rfc = RandomForestClassifier(n_estimators = 7, 
                             criterion = 'entropy', 
                             random_state =7) 
svc = SVC() 
lc = LogisticRegression() 
  
# making predictions on the training set 
for clf in (rfc, knn, svc,lc): 
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_train) 
    print("Accuracy score of ", 
          clf.__class__.__name__, 
          "=",100*metrics.accuracy_score(Y_train,  
                                         Y_pred))

# making predictions on the testing set 
for clf in (rfc, knn, svc,lc): 
    clf.fit(X_train, Y_train) 
    Y_pred = clf.predict(X_test) 
    print("Accuracy score of ", 
          clf.__class__.__name__,"=", 
          100*metrics.accuracy_score(Y_test, 
                                     Y_pred))

"""Conclusion: Random Forest Classifier has the best accuracy for the testing dataset."""












