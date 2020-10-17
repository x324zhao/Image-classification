import numpy as np
import pandas as pd
data=np.genfromtxt('train_data.csv',delimiter=',')
labels=np.genfromtxt('train_labels.csv',delimiter=',')
import numpy as np
#sigmoid function coding
def sigmoid(s):
    return 1 / (1 + np.exp(-s))
#softmax function coding
def softmax(X):
    return np.exp(X)/np.sum(np.exp(X),axis=0)
#cross_entropy loss function
def cross_entropy(p, q):
    return -np.sum(p*np.log(q))
def accuracy(y_true, y_pred):
    if not (len(y_true) == len(y_pred)):
        print('Size of predicted and true labels not equal.')
        return 0.0

    corr = 0
    for i in range(0,len(y_true)):
        corr += 1 if (y_true[i] == y_pred[i]).all()else 0

    return corr/len(y_true)
#learning rate=0.5
lr=0.5
# 784 features added one X0=1 bias term, in totall 785
feature_dimension=785
# number of hidden_layer_neurons
hidden_layer_dimension=40
# number of output neurons
output_layer_dimension=4
# training data : testing data = 4:1
training_data=data[:14852,:]
validation_data=data[14852:19803,:]
testing_data=data[19803:,:]
# training labels :testing labels = 4:1 as well
training_labels=labels[:14852,:]
validation_labels=labels[14852:19803,:]
testing_labels=labels[19803:,:]

# add bias term
training_data_bias=np.c_[np.ones((np.shape(training_data)[0],1)),training_data]
#  Weight 1 matrix between input and hidden layer
w1 = 0.01 * np.random.rand(hidden_layer_dimension, feature_dimension)  # 40 *   785

#  Weight 2 matrix between hidden layer and output layer
w2 = 0.01 * np.random.rand(output_layer_dimension, hidden_layer_dimension)  # 4   *  40

# using 2000 epoches meaning goes through all training data 2000 times
previous_loss=0
for x in range(2000):
    print("Epoch number: ", x)
    # Feed forward
    # to hidden layer
    hidden_out = sigmoid(np.dot(w1, training_data_bias.T))  # 40 * 785      785  *  19803  =   40 * 19803
    # to output layer
    output_layer = softmax(np.dot(w2, hidden_out))  # 4  *   40      40* 19083   =  4 * 19803
    # Backpropagation
    # output - target
    O_T = output_layer.T - training_labels  # 4  * 19803
    # derivative of hidden layer which uses sigmoid function
    dh = hidden_out * (1 - hidden_out)  # 40 * 19803
    O_T_w2_h = np.dot(O_T, w2) * dh.T
    # change of w2 for each step
    dw2 = (np.dot(hidden_out, O_T)).T  # 40 *  19803    19803 * 4 = 40 * 4
    # change of w1 for each step
    dw1 = np.dot(O_T_w2_h.T, training_data_bias)
    # updated w2
    w2 = w2 - lr * dw2 / 14852
    # update w1
    w1 = w1 - lr * dw1 / 14852
    # validation process to avoid overfitting
    validation_data_bias = np.c_[np.ones((np.shape(validation_data)[0], 1)), validation_data]
    validation_data_hidden_out = sigmoid(np.dot(w1, validation_data_bias.T))  # 40 * 785      785  *  19803  =   40 * 19803
    # to output layer
    validation_data_output_layer = softmax(np.dot(w2, validation_data_hidden_out))  # 4  *   40      40* 19083   =  4 * 19803
    loss = cross_entropy(validation_labels, validation_data_output_layer.T)
    current_loss=loss
    print(loss)

    if (abs((current_loss-previous_loss))<0.00001):
        break
    else:
        previous_loss=current_loss
#feed forward for testing data same to triaing data
testing_data_bias=np.c_[np.ones((np.shape(testing_data)[0],1)),testing_data]
testing_hidden_out=sigmoid(np.dot(w1,testing_data_bias.T))
testing_before_softmax=np.dot(w2,testing_hidden_out)
testing_output_layer=softmax(testing_before_softmax)
# one hot coding
predict= []
for i in testing_output_layer.T:
    predicted_labels = [0, 0, 0, 0]
    predicted_labels[np.argmax(i)] = 1
    predict.append(predicted_labels)
y_pred=np.array(predict)
print(accuracy(testing_labels, y_pred))