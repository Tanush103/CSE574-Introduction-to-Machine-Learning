import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    return 1.0 / (1.0 + np.exp(-1.0 * z))
    return  # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    train_set = np.zeros(shape=(50000, 784))
    validation_set = np.zeros(shape=(10000, 784))
    test_set = np.zeros(shape=(10000, 784))
    train_label = np.zeros(shape=(50000,))
    validation_label = np.zeros(shape=(10000,))
    test_label = np.zeros(shape=(10000,))

    train_length = 0
    validation_length = 0
    test_length = 0
    train_label_length = 0
    test_label_length=0
    validation_label_length = 0

    for key in mat:
        
        if "train" in key:
            label = key[-1] 
            a = mat.get(key) #a
            b = range(a.shape[0]) #b
            c = np.random.permutation(b) #c
            length1 = len(a)  ##length1
            length2 = length1-1000 ##length2

            train_set[train_length:train_length + length2] = a[c[1000:], :]
            train_length += length2

            train_label[train_label_length:train_label_length + length2] = label
            train_label_length += length2

            validation_set[validation_length:validation_length + 1000] = a[c[0:1000], :]
            validation_length += 1000

            validation_label[validation_label_length:validation_label_length + 1000] = label
            validation_label_length += 1000

        if "test" in key:
            label = key[-1]  
            a = mat.get(key) #a
            b = range(a.shape[0]) #b
            c = np.random.permutation(b) #c
            length1 = len(a)   #length1
           
            test_set[test_length:test_length + length1] = a[c]
            

            test_label[test_length:test_length + length1] = label
            test_length+=length1
      
    train_size = range(train_set.shape[0])
    train_perm = np.random.permutation(train_size)
    data_train = train_set[train_perm]
    data_train = np.double(data_train)
    data_train = data_train / 255.0
    label_train = train_label[train_perm]

    validation_size = range(validation_set.shape[0])
    vali_perm = np.random.permutation(validation_size)
    data_validation = validation_set[vali_perm]
    data_validation = np.double(data_validation)
    data_validation = data_validation / 255.0
    label_validation = validation_label[vali_perm]

    test_size = range(test_set.shape[0])
    test_perm = np.random.permutation(test_size)
    data_test = test_set[test_perm]
    data_test = np.double(data_test)
    data_test = data_test / 255.0
    label_test = test_label[test_perm]

    # Feature selection
    # Your code here.
    pixels=28*28
    columnsnotneeded=[]
    columnsneeded=[]

    x=data_train[0,0]
    for i in range(pixels):
      train=all(a==x for a in data_train[:,i])
      if(train==True):
        columnsnotneeded.append(i)
      elif(train==False):
        columnsneeded.append(i)

    print(columnsneeded)
    train_data=np.delete(data_train,columnsnotneeded,axis=1)
    test_data=np.delete(data_test,columnsnotneeded,axis=1)
    validation_data=np.delete(data_validation,columnsnotneeded,axis=1)





    

    print('preprocess done')

    return train_data, label_train, validation_data, label_validation, test_data, label_test


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    obj=0.0
    gradient_w1=0.0
    gradient_w2=0.0
    size=training_data.shape[0]
    training_data_length=len(training_data)
    bias=np.ones((len(training_data),1))
    training_data=np.append(training_data,bias,1)
    training_data_T=training_data.T
    output1_1=np.dot(w1,training_data_T)
    output1_2=sigmoid(output1_1)
    output_1_T=output1_2.T
    bias1=np.ones((output_1_T.shape[0],1))
    output_1_T=np.append(output_1_T,bias1,1)
    output_2_1=np.dot(w2,output_1_T.T)
    output_2_2=sigmoid(output_2_1)
    output = np.zeros((n_class,training_data_T.shape[1]))

    i=0
    for i in range(len(training_label)):
      out_label=0
      out_label=int(training_label[i])
      output[out_label,i]=1
    obj += np.sum(output * np.log(output_2_2) + (1.0-output) * np.log(1.0-output_2_2))
    #print(obj)
    Output_delta = output_2_2 - output

    Output_delta_1=Output_delta.reshape(n_class,training_data_T.shape[1])
    gradient_w2=np.dot(Output_delta_1,output_1_T)
    output_w2=np.dot(Output_delta.T,w2)

    output_w2 = output_w2[0:output_w2.shape[0], 0:output_w2.shape[1]-1]
    delta_1_1=(1-output1_2)*output1_2*output_w2.T
    Output_delta_2=delta_1_1.reshape(n_hidden,training_data_T.shape[1])
    gradient_w1=np.dot(Output_delta_2,training_data_T.T)

    obj = ((-1)*obj)/size
    random = np.sum(np.sum(w1**2)) + np.sum(np.sum(w2**2))
    
    obj_val = obj + ((lambdaval * random) / (2.0*size))
    
    gradient_w1=(gradient_w1+lambdaval*w1)/size
    gradient_w2=(gradient_w2+lambdaval*w2)/size
    #print(obj_val)

    



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    obj_grad=np.concatenate((gradient_w1.flatten(),gradient_w2.flatten()),0)
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    #labels = np.array([])
    labels=[]
    # Your code here
    for test_data in data:
      bias_test=np.ones(1)
      bias_included_input=np.hstack((test_data,bias_test))
      output=np.dot(w1,bias_included_input)
      output_sigmoid=sigmoid(output)

      bias_included_output=np.hstack((output_sigmoid,bias_test))
      output_1=np.dot(w2,bias_included_output)
      output_1_sigmoid=sigmoid(output_1)
      labels.append(np.argmax(output_1_sigmoid,axis=0))






    labels=np.array(labels)

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 100
# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 5

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 250}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


