'''
CSC 2515 Homework 2 Q3 Code
Collaborators: Zhimao Lin, Bingzhang Zhu
'''

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from scipy.special import logsumexp
from sklearn.model_selection import KFold
np.random.seed(0)


# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A, B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

def calculate_matrix_a(test_datum, x_train, tau):
    # To make the symbol consistent with the formula
    x = test_datum
    x = np.reshape(x, (1, x.shape[0]))
    norm_matrix = l2(x, x_train)
    denominator = np.exp( logsumexp([np.exp( - norm / (2*np.square(tau)) ) for norm in norm_matrix[0, :]]) )

    matrix_a_diagonal = []
    N = x_train.shape[0]
    for i in range(0, N):
        normi = norm_matrix[0, i]
        numerator = np.exp( - normi / (2*np.square(tau)) )
        ai = numerator / denominator
        matrix_a_diagonal.append(ai)

    matrix_a = np.zeros((N, N))
    np.fill_diagonal(matrix_a, matrix_a_diagonal) 

    return matrix_a


# Locally Reweighted Least Squares
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    '''
    Given a test datum, it returns its prediction based on locally weighted regression

    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''

    print(f"Predicting a test data point: {test_datum}")
    
    matrix_a = calculate_matrix_a(test_datum, x_train, tau)

    x_transpose = np.transpose(x_train)

    # [X^TAX + lamdaI]W(Ridge)* = X^TAy     I is d by d identity matrix
    I = np.identity(x_train.shape[1])
    
    rhs = np.matmul( np.matmul(x_transpose, matrix_a), y_train )
    lhs = np.matmul( np.matmul(x_transpose, matrix_a), x_train ) + lam*I
    w_star = np.linalg.solve(lhs, rhs)
    prediction = np.matmul(np.transpose(test_datum), w_star)

    return prediction
    

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(d,1), x_train, y_train, tau) \
                                    for i in range(N_test)])       
        losses[j] = ((predictions.flatten()-y_test.flatten())**2).mean()

    return losses

#to implement
def run_k_fold(x, y, taus, k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    loss_list = []
    j = 0
    number_of_data_in_a_fold = x.shape[0] // k
    for i in range(0, k):
        print(f"Testing data: From [{0+i*number_of_data_in_a_fold}] to [{number_of_data_in_a_fold+i*number_of_data_in_a_fold}]")
        print(f"Training data: From [0] to [{0+i*number_of_data_in_a_fold}] and from [{number_of_data_in_a_fold+i*number_of_data_in_a_fold}] to the end")

        test_index = idx[0+i*number_of_data_in_a_fold : number_of_data_in_a_fold+i*number_of_data_in_a_fold]
        train_index = np.concatenate((idx[ : 0+i*number_of_data_in_a_fold], idx[number_of_data_in_a_fold+i*number_of_data_in_a_fold : ]), axis=0)
        j = j + 1
        print(f"Using the [{j}]th fold as testing data.")
        train_input = x[train_index]
        train_target = y[train_index]
        test_input = x[test_index]
        test_target = y[test_index]
        losses = run_on_fold(test_input, test_target, train_input, train_target, taus)
        loss_list.append(losses)
        
    # This is k by #taus matrix
    loss_matrix = np.array(loss_list)
    loss_matrix = loss_matrix.mean(axis=0)

    return loss_matrix

def main():
    # In this exercise we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)

    losses = run_k_fold(x, y, taus, k=5)

    plt.figure(figsize=(15, 5))
    plt.plot(taus, losses)
    plt.xlabel('Taus')
    plt.ylabel('Mean Squared Error')
    plt.tight_layout()
    plt.show()

    print("min loss = {}".format(losses.min()))

if __name__ == "__main__":
    main()
