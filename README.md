# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
 import pandas as pd
 import numpy as np
 import seaborn as sns
 from sklearn.model_selection import train_test_split
 from sklearn.neighbors import KNeighborsClassifier
 from sklearn.metrics import accuracy_score, confusion_matrix
 
 data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
 
 data

 <img width="1515" height="486" alt="image" src="https://github.com/user-attachments/assets/4dffd436-7f32-48a9-8d41-ff346e7d7d1d" />

 data.isnull().sum()

 <img width="260" height="552" alt="image" src="https://github.com/user-attachments/assets/d908461d-cbdc-4a52-afb9-d18e16cc532e" />

 missing=data[data.isnull().any(axis=1)]
 missing 

 <img width="1477" height="463" alt="image" src="https://github.com/user-attachments/assets/32269bb4-743e-4a09-855d-d02695bf98b6" />

 data2=data.dropna(axis=0)
 data2

 <img width="1513" height="480" alt="image" src="https://github.com/user-attachments/assets/aa683479-e794-4f25-a282-ad72babb6154" />

 sal=data["SalStat"]
 data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
 print(data2['SalStat'])

 <img width="1259" height="370" alt="image" src="https://github.com/user-attachments/assets/ae319242-6647-4d9e-9cc9-304dddc40c37" />

 sal2=data2['SalStat']
 dfs=pd.concat([sal,sal2],axis=1)
 dfs

 <img width="354" height="473" alt="image" src="https://github.com/user-attachments/assets/195124c5-d33b-4775-8901-ff89de83678a" />

 data2

 <img width="1387" height="463" alt="image" src="https://github.com/user-attachments/assets/25c4f780-ed40-42a6-9907-ac55737669fb" />

 new_data=pd.get_dummies(data2, drop_first=True)
 new_data

 <img width="1747" height="525" alt="image" src="https://github.com/user-attachments/assets/1d74602b-0a70-43dd-aee3-66ba7e23b307" />

columns_list=list(new_data.columns)
print(columns_list)

<img width="1742" height="34" alt="image" src="https://github.com/user-attachments/assets/70e4505e-c298-4ddc-94c2-bb3c9db54b8c" />

 features=list(set(columns_list)-set(['SalStat']))
 print(features)

 <img width="1747" height="43" alt="image" src="https://github.com/user-attachments/assets/9227d98e-883c-4fdd-86e6-34f9a20d8de1" />

 y=new_data['SalStat'].values
 print(y)

 <img width="179" height="42" alt="image" src="https://github.com/user-attachments/assets/ed75f6ce-2a4c-4420-b8bb-6fcf910ce40c" />

 x=new_data[features].values
 print(x)

 <img width="365" height="154" alt="image" src="https://github.com/user-attachments/assets/1e12030e-86bb-441e-ad11-090ad0108bb6" />

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)

<img width="269" height="80" alt="image" src="https://github.com/user-attachments/assets/482bde88-0698-4234-a4ba-e488ed117dc5" />

 prediction=KNN_classifier.predict(test_x)
 confusionMatrix=confusion_matrix(test_y, prediction)
 print(confusionMatrix)

 <img width="131" height="56" alt="image" src="https://github.com/user-attachments/assets/996d755d-b839-492c-8489-c351f0e9dc43" />

 accuracy_score=accuracy_score(test_y,prediction)
 print(accuracy_score)

 <img width="197" height="37" alt="image" src="https://github.com/user-attachments/assets/d7b46f7d-61b0-4426-acdb-ba5edfd4610c" />

print("Misclassified Samples : %d" % (test_y !=prediction).sum())

<img width="262" height="39" alt="image" src="https://github.com/user-attachments/assets/b5e08512-ef1c-4e41-8cda-e23a8ace8460" />

data.shape

<img width="114" height="39" alt="image" src="https://github.com/user-attachments/assets/f8e74af2-693f-462e-9639-ceb136346994" />

import pandas as pd
 from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
 data={
 'Feature1': [1,2,3,4,5],
 'Feature2': ['A','B','C','A','B'],
 'Feature3': [0,1,1,0,1],
 'Target'  : [0,1,1,0,1]
 }
 df=pd.DataFrame(data)
 x=df[['Feature1','Feature3']]
 y=df[['Target']]
 selector=SelectKBest(score_func=mutual_info_classif,k=1)
 x_new=selector.fit_transform(x,y)
 selected_feature_indices=selector.get_support(indices=True)
 selected_features=x.columns[selected_feature_indices]
 print("Selected Features:")
 print(selected_features)

 <img width="1745" height="96" alt="image" src="https://github.com/user-attachments/assets/e6cb2f79-21e3-4a23-8ed2-9eb20b2b35e8" />

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()

<img width="479" height="220" alt="image" src="https://github.com/user-attachments/assets/7762f6a0-19ec-462d-a875-fe9737011e52" />

 tips.time.unique()

 <img width="405" height="56" alt="image" src="https://github.com/user-attachments/assets/5c0fee7c-6105-428f-a097-e5ea9b6ebb57" />

contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)

<img width="200" height="94" alt="image" src="https://github.com/user-attachments/assets/12cbaf06-5277-4030-a24f-42e4add69dda" />

 chi2,p,_,_=chi2_contingency(contingency_table)
 print(f"Chi-Square Statistics: {chi2}")
 print(f"P-Value: {p}")

 <img width="369" height="57" alt="image" src="https://github.com/user-attachments/assets/374c9aeb-fdad-4a8c-9999-f276c2208a84" />

# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the data to a file is been executed
