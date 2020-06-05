import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot  as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


df_test=pd.read_csv("test.csv")
df_train=pd.read_csv("train.csv")

#print(df_test.shape)
#print(df_train.shape)

#print(df_train.head)



df_train['Gender']=df_train['Gender'].fillna(method="bfill")
df_train['Married']=df_train['Married'].fillna(method="bfill")
df_train['Dependents']=df_train['Dependents'].fillna(method="bfill")
df_train['Loan_Amount_Term']=df_train['Loan_Amount_Term'].fillna(df_train['Loan_Amount_Term'].mean())
df_train['LoanAmount']=df_train['LoanAmount'].fillna(df_train['LoanAmount'].mean())
df_train['Self_Employed']=df_train['Self_Employed'].fillna(method="bfill")
df_train['Credit_History']=df_train['Credit_History'].fillna(method="bfill")

total=df_train.isnull().sum().sort_values(ascending=False)
percent=(df_train.isnull().sum()/df_train.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([total,percent],axis=1,keys=['Total','Percent'])
print(missing_data.head(20))


sns.countplot(y='Gender', hue="Loan_Status",data=df_train)
plt.show()


sns.countplot(y="Married", hue="Loan_Status",data=df_train)
plt.show()

sns.countplot(y="Self_Employed",hue="Loan_Status",data=df_train)
plt.show()

sns.countplot(y="Credit_History", hue="Loan_Status",data=df_train)
plt.show()

sns.countplot(y="Property_Area",hue="Loan_Status", data=df_train)
plt.show()


code_numeric = {'Male': 1, 'Female': 2,
'Yes': 1, 'No': 2,
'Graduate': 1, 'Not Graduate': 2,
'Urban': 3, 'Semiurban': 2,'Rural': 1,
'Y': 1, 'N': 0,
'3+': 3}
df_train = df_train.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)
df_test = df_test.applymap(lambda s: code_numeric.get(s) if s in code_numeric else s)
#drop the uniques loan id
df_train.drop('Loan_ID', axis = 1, inplace = True)



Dependents_ = pd.to_numeric(df_train.Dependents)
Dependents__ = pd.to_numeric(df_test.Dependents)
df_train.drop(['Dependents'], axis = 1, inplace = True)
df_test.drop(['Dependents'], axis = 1, inplace = True)
df_train = pd.concat([df_train, Dependents_], axis = 1)
df_test = pd.concat([df_test, Dependents__], axis = 1)

sns.heatmap(df_train.corr())
plt.show()


y = df_train['Loan_Status']
X = df_train.drop('Loan_Status', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

model=LogisticRegression()
model.fit(X_train,y_train)
ypred=model.predict(X_test)
evaluation=f1_score(y_test,ypred)
print(ypred)
print(evaluation)





tree=DecisionTreeClassifier()
tree.fit(X_train,y_train)
ypred_tree=tree.predict(X_test)
print(ypred_tree)
evaluation_tree=f1_score(y_test,ypred_tree)
print(evaluation_tree)


forest=RandomForestClassifier()
forest.fit(X_train,y_train)
ypred_forest=forest.predict(X_test)
print(ypred_forest)
evaluation_forest=f1_score(y_test,ypred_forest)
print(evaluation_forest)
