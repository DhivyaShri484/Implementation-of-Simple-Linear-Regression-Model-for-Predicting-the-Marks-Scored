# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the needed packages
2.Assigning hours To X and Scores to Y
3.Plot the scatter plot
4.Use mse,rmse,mae formmula to find.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: B DHIVYA SHRI
RegisterNumber:  212221230009

import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/Placement_Data.csv')
print(dataset.iloc[3])

print(dataset.iloc[0:4])

print(dataset.iloc[:,1:3])

#implement a simple regression model for predicting the marks scored by students
import pandas as pd
import numpy as np
dataset=pd.read_csv('/content/student_scores.csv')

#implement a simple regression model for predicting the marks scored by students
#assigning hours to X& Scores to Y
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values
print(X)
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

Y_pred=reg.predict(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error

plt.scatter(X_train,Y_train,color="green")
plt.plot(X_train,reg.predict(X_train),color='red')
plt.title("Traning set (H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,reg.predict(X_test),color="pink")
plt.title("Test set (H vs S)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print("MES = ",mse)

mae=mean_absolute_error(Y_test,Y_pred)
print("MAE = ",mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
*/
```

## Output:
<img width="886" alt="2022-10-09 (12)" src="https://user-images.githubusercontent.com/94505585/194763069-8080b95e-89c0-4b6e-9179-bbdb8dcd9abc.png">
<img width="889" alt="2022-10-09 (14)" src="https://user-images.githubusercontent.com/94505585/194763080-d57f5005-549a-470a-8f10-dbdcece2ac92.png">
<img width="903" alt="2022-10-09 (15)" src="https://user-images.githubusercontent.com/94505585/194763086-8c4c0153-cf34-4f26-a526-9eb47097f10a.png">
<img width="902" alt="2022-10-09 (16)" src="https://user-images.githubusercontent.com/94505585/194763092-ef080004-7348-4a52-b550-41c356c70aed.png">
<img width="904" alt="2022-10-09 (17)" src="https://user-images.githubusercontent.com/94505585/194763098-dd79d2ae-0106-4599-9530-11eadbbb5080.png">


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
