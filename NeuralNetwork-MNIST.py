#!/usr/bin/env python
# coding: utf-8

# In[119]:


STUDENT_NAME = "Aakanksha_Rani" #Put your name
STUDENT_ROLLNO = "MT2022001" #Put your roll number
CODE_COMPLETE = True
# set the above to True if you were able to complete the code
# and that you feel your model can generate a good result
# otherwise keep it as False
# Don't lie about this. This is so that we don't waste time with
# the autograder and just perform a manual check
# If the flag above is True and your code crashes, that's
# an instant deduction of 2 points on the assignment.
#


# In[120]:


#@PROTECTED_1_BEGIN
## No code within "PROTECTED" can be modified.
## We expect this part to be VERBATIM.
## IMPORTS 
## No other library imports other than the below are allowed.
## No, not even Scipy
import numpy as np 
import pandas as pd 
import sklearn.model_selection as model_selection 
import sklearn.preprocessing as preprocessing 
import sklearn.metrics as metrics 
from tqdm import tqdm # You can make lovely progress bars using this


# In[121]:


## FILE READING: 
## You are not permitted to read any files other than the ones given below.
X_train = pd.read_csv("train_X.csv",index_col=0).to_numpy()
y_train = pd.read_csv("train_y.csv",index_col=0).to_numpy().reshape(-1,)
X_test = pd.read_csv("test_X.csv",index_col=0).to_numpy()
submissions_df = pd.read_csv("sample_submission.csv",index_col=0)
#@PROTECTED_1_END


# In[122]:


X_train.shape


# In[123]:


y_train.shape


# In[124]:


X_test.shape


# In[125]:


submissions_df.shape


# **Analyzing the Dataset**

# In[126]:


np.unique(X_train)


# In[128]:


np.unique(y_train)


# In[130]:


#Counting occurences of each digits
y=np.zeros((1,10))
for count in range (10):
    print("occurance of ",count,"=",np.count_nonzero(y_train==count))
    y[0,count-1]= np.count_nonzero(y_train==count)


# In[131]:


def normalize_input(X1,X2):
    return X1 / 255.0, X2 / 255.0


# In[132]:


X_train, X_test = normalize_input(X_train ,X_test)


# In[133]:


# function to split train and test dataset
# train dataset=80% , test dataset=30%
def train_test_split(x,y,split_value):
    nt=(split_value*x.shape[0])//100
    
    X_train_s=x[:nt,:]
    Y_train_s=y[:nt]
    
    X_test_s=x[nt:,:]
    Y_test_s=y[nt:]
    
    return X_train_s,X_test_s,Y_train_s,Y_test_s


# In[134]:


X_train_s,X_test_s,Y_train_s,Y_test_s=train_test_split(X_train,y_train,split_value=80)


# In[135]:


print(X_train_s.shape)
print(X_test_s.shape)
print(Y_train_s.shape)
print(Y_test_s.shape)


# In[136]:


#converting y_train_s in one hot encoder representation 
Y_train_s1=np.zeros((10,Y_train_s.shape[0]))
for column in range (Y_train_s.shape[0]):
    value=Y_train_s[column]
    for row in range (10):
        if (value==row):
            Y_train_s1[value,column]=1


# In[137]:


print(Y_train_s.shape)
print(Y_train_s1.shape)


# **Initializing Weights and Bias**

# In[138]:


#function to initialize the parameters - weights and bias
#only one hidden layer
def initialize_parameters():
    W1 = np.random.rand(10, 784)*0.01
    B1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10)*0.01
    B2 = np.random.rand(10, 1) - 0.5
    return W1,B1,W2, B2


# **Various Activation Functions**

# In[139]:


#RELU
def relu(Z): #for hidden layers
    A=np.maximum(0,Z)
    return A

def relu_der(dA):
    return dA>0
    
#tan h 
def tanh(Z): #for hidden layers
    A1=np.exp(Z)-np.exp(-Z)
    A2=np.exp(Z)+np.exp(-Z)
    A=A1/A2
    return A

def tanh_der(dA):
    A1=np.exp(dA)-np.exp(-dA)
    A2=np.exp(dA)+np.exp(-dA)
    A=A1/A2
    dZ=1-A*A
    return dZ

#Sigmoid
def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def sigmoid_der(dA):
    t = 1/(1+np.exp(-dA))
    dZ = dA*t * (1-t)
    return dZ

#Softmax
def softmax(Z): #for output layer in multiclass classification
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    


# **Forward Propagation**

# In[140]:


def forward(W1, B1, W2, B2, X):
    Z1 = np.dot(W1,X) + B1
    A1=linear_activation_forward(Z1,"relu")
    #A1=linear_activation_forward(Z1,"tanh") 
    Z2 = np.dot(W2,A1) + B2
    A2=linear_activation_forward(Z2,"softmax")
    return Z1, A1, Z2, A2


# In[141]:


def linear_activation_forward(Z, activation):
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "relu":
        A = relu(Z) 
    elif activation == "softmax":
        A = softmax(Z)
    elif activation == "tanh":
        A = tanh(Z)
    
    return A


# In[142]:


def shape_of_input(X):
    return X.shape[1]


# **Backward Propagation**

# In[143]:


def backward(Z1, A1, Z2, A2, W1, W2, X, Y):
   dZ2 = A2 - Y_train_s1
   x=shape_of_input(X)
   dW2 = 1 / x * np.dot(dZ2,A1.T)
   dB2 = 1 / x * np.sum(dZ2)
   dZ1 = np.dot(W2.T,dZ2) * relu_der(Z1)
   dW1 = 1 / x * np.dot(dZ1,X.T)
   dB1 = 1 / x * np.sum(dZ1)
   return dW1, dB1, dW2, dB2


# 
# 

# **Update Parameters**
# 
# * **Updated Weights
# W=W-(learning_rate* dW)**
# 
# * **Updated Bias
# B=B-(learning_rate* db)**

# In[144]:


def update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2, learning_rate):
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1    
    W2 = W2 - learning_rate * dW2  
    B2 = B2 - learning_rate * dB2    
    return W1, B1, W2, B2


# **Function to get accuracy**

# In[160]:


def get_accuracy(predictions, Y):
    print("Y_Predicted: ",predictions," Y_Actual: ", Y)
    temp=np.sum(predictions == Y) / Y.shape[0]
    return temp * 100 #out of 100 percent


# **Function to call sub functions (Gradient descent)**

# In[146]:


def one_hidden_layer_model(X, Y, learning_rate, epoch):
    print("Training ")
    W1, B1, W2, B2 = initialize_parameters()
    print("Total iterations :",epoch)
    for ix in range(epoch):
        Z1, A1, Z2, A2 = forward(W1, B1, W2, B2, X)
        dW1, dB1, dW2, dB2 = backward(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, B1, W2, B2 = update_parameters(W1, B1, W2, B2, dW1, dB1, dW2, dB2,learning_rate)
        if ix % 200 == 0:
            print("Iteration : ", ix)
            predictions=np.argmax(A2, 0)
            print(get_accuracy(predictions, Y))
    print("Training done")
    return W1, B1, W2, B2


# In[147]:


# def reshape_of_matrix(X):
#     temp=np.reshape(X,[784,X.shape[0]])
#     return temp


# In[148]:


X_train_s=np.transpose(X_train_s)
#X_train_s=reshape_of_matrix(X_train_s)


# In[149]:


W1, B1, W2, B2 = one_hidden_layer_model(X_train_s, Y_train_s, 0.68, 1000)


# In[150]:


X_test_s=np.transpose(X_test_s)
#X_test_s=reshape_of_matrix(X_test_s)


# In[151]:


def make_prediction(W1, B1, W2, B2,X):
    t1, t2, t3, A2 = forward(W1, B1, W2, B2, X)
    return np.argmax(A2, 0)
     


# In[161]:


Y_test_s_pred = make_prediction( W1, B1, W2, B2,X_test_s)
get_accuracy(Y_test_s_pred, Y_test_s)


# Tried with one hidden layer
# 
# Activation function at hidden layer - ReLU (max accuracy achieved-93.55 %)
# 
# Activation function at output layer - Softmax 
# 
# *also tried using tanh in hidden layer

# In[164]:


X_test=np.transpose(X_test)


# In[170]:


Y_predicted = make_prediction( W1, B1, W2, B2,X_test)
submissions_df=pd.DataFrame({'label':Y_predicted})


# In[171]:


submissions_df.head()


# In[167]:


submissions_df.shape


# In[169]:


#@PROTECTED_2_BEGIN 
##FILE WRITING:
# You are not permitted to write to any file other than the one given below.
submissions_df.to_csv("{}__{}.csv".format(STUDENT_ROLLNO,STUDENT_NAME))
#@PROTECTED_2_END


# In[ ]:




