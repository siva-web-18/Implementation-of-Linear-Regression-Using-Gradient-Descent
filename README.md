# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries 
2.Load the Dataset 
3.Select Input and Output Variables 
4.Split Data into Training & Testing Sets 
5.Feature Scaling (Optional)

## Program:
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
 X=np.c_[np.ones(len(X1)),X1]

 theta=np.zeros(X.shape[1]).reshape(-1,1)

 for _ in range(num_iters):
 #Calculate predictions
 predictions=(X).dot(theta).reshape(-1,1)

 #calculate errors
 errors=(predictions-y).reshape(-1,1)
 #Update theta using gradient descent
 theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
68
 return theta
data=pd.read_csv("C:/Users/hp15cs1000/Downloads/50_Startups.csv",header=None)
data.head()
#Assuming the last column is your target variable 'y'
#and the preceding columns are your features 'X'
X=(data.iloc[1:,:-2].values)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)
#Learn model Parameters
theta=linear_regression(X1_Scaled,Y1_Scaled)
#Predict target value for a new data point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
/*
Program to implement the linear regression using gradient descent.
Developed by: siva R 
RegisterNumber:  25007668
*/
```

## Output:
![linear regression using gradient descent](sam.png)
<img width="912" height="636" alt="ml git 3 1" src="https://github.com/user-attachments/assets/700f320c-19b1-43f3-b281-59643e641835" />
<img width="1028" height="666" alt="ml git 3 2" src="https://github.com/user-attachments/assets/646c64f9-4406-4f57-9468-1ebb2be2005c" />



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
