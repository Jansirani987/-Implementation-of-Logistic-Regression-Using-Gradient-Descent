# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Prepare your data - Clean and format your data - Split your data into training and testing sets

2.Define your model - Use a sigmoid function to map inputs to outputs - Initialize weights and bias terms

3.Define your cost function - Use binary cross-entropy loss function - Penalize the model for incorrect predictions

4.Define your learning rate - Determines how quickly weights are updated during gradient descent

5.Train your model - Adjust weights and bias terms using gradient descent - Iterate until convergence or for a fixed number of iterations

6.Evaluate your model - Test performance on testing data - Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters - Experiment with different learning rates and regularization techniques

8.Deploy your model - Use trained model to make predictions on new data in a real-world application.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: JANSI RANI A A
RegisterNumber:  212224040130

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('Placement_Data.csv')
print(data.head())

# Drop unnecessary columns
data = data.drop('sl_no', axis=1)
data = data.drop('salary', axis=1)

# Convert categorical columns to category type
data["gender"] = data["gender"].astype('category')
data["ssc_b"] = data["ssc_b"].astype('category')
data["hsc_b"] = data["hsc_b"].astype('category')
data["degree_t"] = data["degree_t"].astype('category')
data["workex"] = data["workex"].astype('category')
data["specialisation"] = data["specialisation"].astype('category')
data["status"] = data["status"].astype('category')
data["hsc_s"] = data["hsc_s"].astype('category')

print(data.dtypes)

# Encode categorical columns
data["gender"] = data["gender"].cat.codes
data["ssc_b"] = data["ssc_b"].cat.codes
data["hsc_b"] = data["hsc_b"].cat.codes
data["degree_t"] = data["degree_t"].cat.codes
data["workex"] = data["workex"].cat.codes
data["specialisation"] = data["specialisation"].cat.codes
data["status"] = data["status"].cat.codes
data["hsc_s"] = data["hsc_s"].cat.codes

print(data.head())

# Features and target
x = data.iloc[:, :-1].values   # all rows, all columns except last
y = data.iloc[:, -1].values    # all rows, last column (target)

# Initialize random weights
theta = np.random.randn(x.shape[1])
Y = y

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Loss function
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

# Gradient descent
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

# Train model
theta = gradient_descent(theta, x, y, alpha=0.01, num_iterations=1000)

# Prediction function
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred

# Predictions
y_pred = predict(theta, x)

# Accuracy
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print(y_pred)

# Test new input
xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print("Prediction for xnew 1:", y_prednew)

xnew = np.array([[0, 0, 0, 0, 0, 2, 8, 2, 0, 0, 1, 0]])
y_prednew = predict(theta, xnew)
print("Prediction for xnew 2:", y_prednew)

*/
```

## Output:
<img width="666" height="139" alt="Screenshot (739)" src="https://github.com/user-attachments/assets/5dfe7884-2d59-4f53-a670-d8c2e69c7300" />
<img width="724" height="130" alt="Screenshot (740)" src="https://github.com/user-attachments/assets/2e68af52-0dd1-4d4f-909a-f56998fc8a9a" />
<img width="719" height="137" alt="Screenshot (741)" src="https://github.com/user-attachments/assets/bf9ae1d3-c6ac-4879-b921-9a2cd35a8f1a" />
<img width="410" height="138" alt="Screenshot (742)" src="https://github.com/user-attachments/assets/896e0566-4db3-49a9-879a-5d70091d6e46" />
<img width="730" height="205" alt="Screenshot (743)" src="https://github.com/user-attachments/assets/72df6cdc-1150-42d0-b76d-604f20e2ad8b" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

