#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jun 12 11:08:52 2017

@author: celinejin
"""

import numpy as np

def matrix_factorization(R,P,Q,K, steps = 5000, alpha=0.0002, beta=0.02):
    '''
    Parameters
    ----------
    R : numpy array
        rating matrix.
    P : numpy array
        |U|*K (user features matrix).
    Q : numpy array
        |D|*K (item features matrix).
    K : integer
        latent features.
    steps : integer
        iterations. The default is 5000.
    alpha : float
        learning rate. The default is 0.0002.
    beta : float
        regularization parameter. The default is 0.02.

    Returns
    -------
    Two matrix embeddings P and Q transpose.

    '''
    Q = Q.T
    
    for step in range(steps):
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    # calculate error
                    eij = R[i][j] - np.dot(P[i,:],Q[:,j])
                    
                    for k in range(K):
                        # calculate gradient with a and beta parameter
                        P[i][k] = P[i][k] + alpha * (2*eij * Q[k][j] - beta*P[i][k])
                        Q[k][j] = Q[k][j] + alpha * (2*eij * P[i][k] - beta*Q[k][j])
        Rhat = np.dot(P,Q)
        
        error = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i][j] > 0:
                    error += pow(R[i][j]-np.dot(P[i,:],Q[:,j]),2)
                    for k in range(K):
                        error += (beta/2) * (pow(P[i][k],2)+pow(Q[k][j],2))
                        
        # set 0.001 as local minimum
        if error < 0.001:
            break
    return P, Q.T

### Toy example
R = [[7,3,0,1],[4,0,0,1],[1,1,0,5],[1,0,0,4],[0,1,0,5],[2,0,0,3]]

R = np.array(R)
N = len(R) # N: number of users
M = len(R[0]) # M: number of items
K = 3 # number of features, need fine-tune

P = np.random.rand(N,K)
Q = np.random.rand(M,K)

nP, nQ = matrix_factorization(R,P,Q,K)

nR = np.dot(nP,nQ.T) 
print(R)
print(nR)
