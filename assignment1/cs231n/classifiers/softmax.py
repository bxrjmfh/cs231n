from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    D = X.shape[1]
    C = W.shape[1]
    for i, x_i in enumerate(X):
        f_i = np.dot(x_i, W)
        # f_i is Cx1 size
        # L_i = -log(\frac{e^{f_{yi}}}{\sum e^{f_j}})
        f_i -= f_i.max()
        # to avoid unstable result
        e_fi = np.exp(f_i)
        L_i = -np.log(e_fi[y[i]] / e_fi.sum())
        loss += L_i
        dW += np.dot(x_i.reshape(D,-1),e_fi.reshape(-1,C))/np.sum(e_fi)
        # minus every column
        dW[:, y[i]] -= x_i
    loss /= len(X)
    loss += reg * np.sum(W*W)
    dW /= len(X)
    dW += 2* reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    D = X.shape[1]
    C = W.shape[1]
    N = X.shape[0]
    f_ij = np.dot(X,W)
    # f_ij is the output with shape NxC
    f_ij-=f_ij.max()
    e_ij = np.exp(f_ij)
    # L = np.array([-np.log(e_ij[i][y[i]] / e_ij[i].sum())
    #               for i in range(e_ij.shape[0]) ])
    L = -np.log(e_ij[range(N),list(y)]/e_ij.sum(axis=1))
    # L with shape Nx1
    loss = L.sum()/N + reg * np.sum(W * W)
    dS = e_ij.copy()/e_ij.sum(axis=1).reshape(-1,1)
    dS[range(N),list(y)] -= 1
    dW = np.dot(X.T,dS)/N + 2*reg * W
    # expand operation on matrix
    # usage range and list in index

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
