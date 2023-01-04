import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import copy, math
from sklearn.linear_model import LogisticRegression



dataset = pd.read_csv('/Users/huytuannguyen/Desktop/FPT/My self/MachineLearning/Logistic Regression/Exoplanet Hunting in Deep Space data/exoTrain.csv')
df = dataset.dropna()
x_train = df.drop(['LABEL'], axis=1)
y_train = df['LABEL']

class LogisticRegressionModel():

    def __init__(self):
        self.weights = 0
        self.bias = 0
        self.cost_list = []
        self.epoch_list = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def cost(self, X, y, w, b, lambda_ =1 ):
        """
        Computes the cost over all examples
        Args:
        X : (ndarray Shape (m,n)) data, m examples by n features
        y : (array_like Shape (m,)) target value 
        w : (array_like Shape (n,)) Values of parameters of the model      
        b : scalar Values of bias parameter of the model
        lambda_: unused placeholder
        Returns:
        total_cost: (scalar)         cost 
        """

        m, n = X.shape
        total_cost = 0.0
        for i in range(m):
            z_i = np.dot(X[i],w) + b
            f_wb_i = self.sigmoid(z_i)
            total_cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
                
        total_cost = total_cost / m
        return total_cost
    
    def gradient(self, X, y, w, b):
        """
        Computes the gradient for linear regression 
    
        Args:
        X (ndarray (m,n): Data, m examples with n features
        y (ndarray (m,)): target values
        w (ndarray (n,)): model parameters  
        b (scalar)      : model parameter

        Returns
        dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w. 
        dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b. 
        """
        
        m,n = X.shape
        dj_dw = np.zeros((n,))                           #(n,)
        dj_db = 0.

        for i in range(m):
            f_wb_i = self.sigmoid(np.dot(X[i],w) + b)          #(n,)(n,)=scalar
            err_i  = f_wb_i  - y[i]                       #scalar
            dj_db = dj_db + err_i
            
            for j in range(n):
                dj_dw[j] = dj_dw[j] + err_i * X[i,j]      #scalar
            
        dj_dw = dj_dw/m                                   #(n,)
        dj_db = dj_db/m      
        return dj_db, dj_dw
    
    def gradient_descent(self,X, y, beta, learning_rate):
        y = y.reshape(-1, 1)
        gradients = np.dot(X.T, self.sigmoid(np.dot(X, beta.T)) - y) / len(y)
        new_betas = beta - learning_rate * gradients.T

        return new_betas
    
    def train(self, X, y, epochs, batch_size, learning_rate = 0.01):
        """
        Performs batch gradient descent
        
        Args:
        X (ndarray (m,n)   : Data, m examples with n features
        y (ndarray (m,))   : target values
        alpha (float)      : Learning rate
        epochs (scalar) : number of iterations to run gradient descent
        
        Returns:
        w (ndarray (n,))   : Updated values of parameters
        b (scalar)         : Updated value of parameter 
        """

        # init parameters
        self.weights = np.zeros_like(X.shape[0])  #avoid modifying global w within function
        self.bias = 0
        total_samples = X.shape[0]
        
        if batch_size > total_samples: # In this case mini batch becomes same as batch gradient descent
            batch_size = total_samples
        
        num_batches = int(total_samples/batch_size)

        for i in range(epochs):
            random_indices = np.random.permutation(total_samples)
            # print(type(random_indices))
            # print(type(X))
            X_tmp = X.to_numpy()[random_indices,:]
            y_tmp = y.to_numpy()[random_indices]
            

            for j in range(0,total_samples,batch_size):
                Xj = X_tmp[j:j+batch_size]
                yj = y_tmp[j:j+batch_size]

                y_pred = self.sigmoid(Xj)
                # Calculate the gradient and update the parameters
                dj_db, dj_dw = self.gradient(X, y, self.weights, self.bias)   

                # Update Parameters using w, b, alpha and gradient
                self.weights = self.weights - learning_rate * dj_dw               
                self.bias = self.bias - learning_rate * dj_db            

                cost = self.cost(Xj, yj, self.weights, self.bias)   
                # cost_list.append(cost)

            if i%10 == 0:
                self.cost_list.append(cost)
                self.epoch_list.append(i)
        
            
        # return w, b, cost_list, epoch_list        #return final w,b and J history for graphing

    def predict(self, X, w, b):
        """
        Predict whether the label is 0 or 1 using learned logistic
        regression parameters w
        
        Args:
        X : (ndarray Shape (m, n))
        w : (array_like Shape (n,))      Parameters of the model
        b : (scalar, float)              Parameter of the model

        Returns:
        p: (ndarray (m,1))
            The predictions for X using a threshold at 0.5
        """
        # number of training examples
        m, n = X.shape   
        p = np.zeros(m)
    
        # Loop over each example
        for i in range(m):   

            # Calculate f_wb (exactly how you did it in the compute_cost function above) 
            # using a couple of lines of code
            f_wb = self.sigmoid(np.dot(X[i],w) + b)

            # Calculate the prediction for that training example 
            if f_wb >= 0.5:
                p[i] = 1
            else:
                p[i] = 0
    
        return p

model = LogisticRegressionModel()
model.train(x_train.astype('float128'), y_train.astype('float128'), 120, 100)
