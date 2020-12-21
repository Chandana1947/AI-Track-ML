#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np


# In[54]:


data = pd.read_excel('C:\\Users\\Gaddam Srinivas\\Load Data in kW.xlsx')


# In[55]:


data.isnull().sum()


# In[56]:


data.head()


# In[57]:


X=data[data.columns[:3]]
X.head()


# In[58]:


y=data[['Load (kW)']]
y.head()


# In[75]:


from sklearn.model_selection import train_test_split
#train test split
X_train, X_test, y_train,y_test = train_test_split(x, y, test_size=0.2,random_state=121)


# # ANN implementation using Adagrad Optimizer

# In[76]:


def define_structure(X, Y):
    input_unit = X.shape[0] # size of input layer
    hidden_unit = 4 #hidden layer of size 4
    output_unit = Y.shape[0] # size of output layer
    return (input_unit, hidden_unit, output_unit)
(input_unit, hidden_unit, output_unit) = define_structure(X_train, y_train)
print("The size of the input layer is:  = " + str(input_unit))
print("The size of the hidden layer is:  = " + str(hidden_unit))
print("The size of the output layer is:  = " + str(output_unit))


# In[77]:


def parameters_initialization(input_unit, hidden_unit, output_unit):
    np.random.seed(2) 
    W1 = np.random.randn(hidden_unit, input_unit)*0.01
    b1 = np.zeros((hidden_unit, 1))
    W2 = np.random.randn(output_unit, hidden_unit)*0.01
    b2 = np.zeros((output_unit, 1))
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# In[78]:


def sigmoid(z):
    return 1/(1+np.exp(-z))


# In[79]:


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2}
    
    return A2, cache


# In[80]:


def cross_entropy_cost(A2, Y, parameters):
    # number of training example
    m = Y.shape[1] 
    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1-Y), np.log(1 - A2))
    cost = - np.sum(logprobs) / m
    cost = float(np.squeeze(cost))
                                    
    return cost


# In[81]:


def backward_propagation(parameters, cache, X, Y):
    #number of training example
    m = X.shape[1]
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']
   
    dZ2 = A2-Y
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1/m) * np.dot(dZ1, X.T) 
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1, "db1": db1, "dW2": dW2,"db2": db2}
    
    return grads


# In[89]:


def do_adagrad(parameters,grads,learning_rate = 0.01):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
   
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    v_w1, v_b1, v_w2, v_b2, eps = 0, 0, 0, 0, 1e-8
    for i in range(max_epochs):
        dw1, db1, dw2, db2 = 0, 0, 0, 0
        for x,y in zip(X,Y):
            dw1 += grad_w(W1, b1, x, y)
            db1 += grad_b(W1, b1, x, y)
            dw2 += grad_w(W2, b2, x, y)
            db2 += grad_b(W2, b2, x, y)
            
        v_w1 = v_w1 + dw1**2
        v_b1 = v_b1 + db1**2
        v_w2 = v_w2 + dw2**2
        v_b2 = v_b2 + db2**2
        
        W1 = W1 - (eta/np.sqrt(v_w1 + eps)) * dw1
        b1 = b1 - (eta/np.sqrt(v_b1 + eps)) * db1
        W2 = W2 - (eta/np.sqrt(v_w2 + eps)) * dw2
        b2 = b2 - (eta/np.sqrt(v_b2 + eps)) * db2
        
    parameters = {"W1": W1, "b1": b1,"W2": W2,"b2": b2}
    return parameters


# In[ ]:


def neural_network_model(X, Y, hidden_unit, num_iterations = 1000):
    np.random.seed(3)
    input_unit = define_structure(X, Y)[0]
    output_unit = define_structure(X, Y)[2]
    
    parameters = parameters_initialization(input_unit, hidden_unit, output_unit)
   
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = cross_entropy_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = do_adagrad(parameters, grads)
        if i % 5 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return parameters
parameters = neural_network_model(X_train, y_train, 4, num_iterations=1000)


# In[ ]:


def prediction(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    
    return predictions


# In[ ]:


predictions = prediction(parameters, X_train)
print ('Accuracy Train: %d' % float((np.dot(y_train, predictions.T) + np.dot(1 - y_train, 1 - predictions.T))/float(y_train.size)*100) + '%')
predictions = prediction(parameters, X_test)
print ('Accuracy Test: %d' % float((np.dot(y_test, predictions.T) + np.dot(1 - y_test, 1 - predictions.T))/float(y_test.size)*100) + '%')

