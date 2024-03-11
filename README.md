# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SAKTHI PRIYA D
RegisterNumber: 212222040139
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df = pd.read_csv('/content/student_scores.csv')
df.head()

#segregating data to variables
x = df.iloc[:, :-1].values
x

#splitting train and test data
y = df.iloc[:, -1].values
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#displaying predicted values
y_pred

#displaying actual values
y_test

#graph plot for training data
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="red")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#graph plot for test data
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_train,regressor.predict(x_train),color="green")
plt.title("Hours vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(y_test,y_pred)
print("MSE= ",mse)

mae=mean_absolute_error(y_test,y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

## Output:
df.head()
![image](https://github.com/sakthipriyadhanusu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393194/08fc5efe-9921-4ee6-b20e-d64ebe1076a3)

df.tail()
![image](https://github.com/sakthipriyadhanusu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393194/f0173a1a-81ba-4bc8-9276-aa13895d3d29)

Array value of X
![image](https://github.com/sakthipriyadhanusu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393194/d5b9179f-8ade-4160-ad0d-d32087bd29e8)

Array value of Y
![image](https://github.com/sakthipriyadhanusu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393194/866f08f1-00c3-4d58-9f0f-44bce34e8ac0)

Values of y preidction
![image](https://github.com/sakthipriyadhanusu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393194/47266174-7ed8-404f-a488-2de2c3fe1466)

Array values of Y test
![image](https://github.com/sakthipriyadhanusu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393194/966ea121-2047-4077-8ea9-46e9b874a012)

Training and Testing set
![image](https://github.com/sakthipriyadhanusu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393194/9d75a309-e387-4c21-8071-2a16a946e0a0)

![image](https://github.com/sakthipriyadhanusu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393194/d9e2945e-808a-4ef7-911f-7af6a0eddf16)

Values of MSE,MAE and RMSE
![image](https://github.com/sakthipriyadhanusu/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119393194/42bb0ec8-24ce-48be-9612-aa4c0dec772a)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
