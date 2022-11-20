from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
from tqdm import tqdm
from datetime import datetime


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    # 10
    num_train = X.shape[0]
    loss = 0.0
    for i in tqdm(range(num_train)):
        scores = X[i].dot(W)
        # (D,)·(D,C) -> (C,)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, j] += X[i].T
                dW[:, y[i]] += -X[i].T
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += 0.5 * reg * np.sum(W * W)
    # W*W denotes every element's square (point multiply each vector by itself)
    # and np.sum sum them all
    # this denotes the level of regularization
    dW += reg * W
    # add the loss to regularity item

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****=
    num_train = X.shape[0]
    num_classes = W.shape[1]
    scores = np.matmul(X, W)
    # the result is all s_ij matrix (NxC) , the same result as X.dot(W)
    correct_scores = scores[range(scores.shape[0]), list(y)].reshape(-1, 1)
    # the correct score s_yi
    margins = np.maximum(0, scores - correct_scores + 1)
    # calculate margins in form of matrix
    # using the np.maximun function as the max operation
    margins[range(num_train), list(y)] = 0
    # set the yi column be zero
    loss = np.sum(margins) / num_train + 0.5 * reg * np.sum(W * W)

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # bullshit
    # we calculate the num of non-zero margins as
    # mask = np.where(margins>0,True,False)
    # # mask have shape (D,C)
    # k = np.sum(mask.astype("int"),axis=0).reshape(-1,1)
    # # dw_j = xxx -k*X_yi , k have shape 10000x1
    #
    # # select the X_i to be plus
    # x_1 = np.dot(X,mask.T)
    # # x_1 is
    #
    # #  select the X_yi to be minus
    # x_2 = k*X
    # # calculate the scalar 10000,D
    # x_temp = x_2[np.newaxis,...].repeat(20,axis=0)
    # # expand k*x_yi to CxNxD
    #
    # # turn the label to one-hot code
    # y_onehot = np.zeros((y.shape[0],y.max()+1))
    # y_onehot[range(y.shape[0]),y]=1
    # # y_onehot is NxC
    # x_2 =
    #### shit

    coeff_mat = np.zeros((num_train, num_classes))
    # make a mask that is NxC
    coeff_mat[margins > 0] = 1
    coeff_mat[range(num_train), list(y)] = 0
    # set the corresponding yi mat to zero
    coeff_mat[range(num_train), list(y)] = -np.sum(coeff_mat, axis=1)
    # set the corresponding yi mat to -k*x_yi

    dW = (X.T).dot(coeff_mat)
    dW = dW / num_train + reg * W

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


import math
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np


def division_10exponential_to_k(low, high, k):
    """
    :param low: the lower bounder of list
    :param high: the higher bounder of list
    :param k: divide the list to k fold
    :return: a splited list
    """
    low_e = math.log10(low)
    high_e = math.log10(high)
    delta = (high_e - low_e) / k
    return 10 ** np.arange(low_e, high_e, delta)


def visualize_lr_reg_valloss(lr_range, reg_range, results, iter):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.log10(lr_range)
    Y = np.log10(reg_range)
    X, Y = np.meshgrid(X, Y)
    val_res = np.array([r[1] for r in results]).reshape(X.shape)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, val_res, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(val_res.min() - 0.01, val_res.max() + 0.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    # current dateTime
    now = datetime.now()

    # convert to string
    date_time_str = now.strftime("%Y-%m-%d %H:%M:%S")
    plt.savefig(date_time_str + 'valacc_of_%d_th_:_%f.png' % (iter, val_res.max()))
    plt.show()


def generation_space(lr_range, reg_range, results, k_fold, top_k):
    record = np.copy(results)
    results = np.array(results)[:, 1]
    # select the validation result
    top_k_index_1d = np.argsort(results)[-top_k:]
    # find index
    best_record = {(lr_range[top_k_index_1d[-1] // len(reg_range)],
                 reg_range[top_k_index_1d[-1] % len(reg_range)])
                : list(record[top_k_index_1d[-1]])}
    print('best'+str(best_record))

    # if best_res > 0.4:
    #     return [(lr_range[top_k_index_1d[-1] // len(lr_range)],
    #              reg_range[top_k_index_1d[-1] % len(reg_range)])]
    #   return the exactly point

    #   when maxium on broader , expand the broader
    #   reg_flag = -1 : extenfd the lower column bound
    #              1 : extend the upper column bound
    #
    if 0 == top_k_index_1d[-1] % len(reg_range):
        reg_flag = -1
    elif len(reg_range) - 1 == top_k_index_1d[-1] // len(reg_range):
        reg_flag = 1
    else:
        reg_flag = 0

    if 0 == top_k_index_1d[-1] // len(reg_range):
        lr_flag = -1
    elif len(lr_range) - 1 == top_k_index_1d[-1] // len(reg_range):
        lr_flag = 1
    else:
        lr_flag = 0

    # adjust searching space
    print("adjust gird from (%s,%s) -> (%s,%s)" % (lr_range.min(), reg_range.min(), lr_range.max(), reg_range.max()))
    if lr_flag != 0 and reg_flag != 0:
        # expand and return
        print("expand gird ")
        delta_lrexpo = (math.log10(lr_range[-1]) - math.log10(lr_range[0])) / 2
        delta_regexpo = (math.log10(reg_range[-1]) - math.log10(reg_range[0])) / 2

        if lr_flag == -1:
            lr_range = division_10exponential_to_k(low=lr_range[0] * (10 ** (-delta_lrexpo)),
                                                   high=lr_range[0] * (10 ** (delta_lrexpo)),
                                                   k=k_fold)
        elif lr_flag == 1:
            lr_range = division_10exponential_to_k(low=lr_range[-1] * (10 ** (-delta_lrexpo)),
                                                   high=lr_range[-1] * (10 ** (delta_lrexpo)),
                                                   k=k_fold)

        if reg_flag == -1:
            reg_range = division_10exponential_to_k(low=reg_range[0] * (10 ** (-delta_regexpo)),
                                                    high=reg_range[0] * (10 ** (delta_regexpo)),
                                                    k=k_fold)
        elif reg_flag == 1:
            reg_range = division_10exponential_to_k(low=reg_range[-1] * (10 ** (-delta_regexpo)),
                                                    high=reg_range[-1] * (10 ** (delta_regexpo)),
                                                    k=k_fold)

    else:
        # reduce the gird
        # to find the top_k index
        print("decrease area")
        lr_range = lr_range[top_k_index_1d // len(reg_range)]
        print(lr_range)
        reg_range = reg_range[top_k_index_1d % len(reg_range)]
        print(reg_range)
        lr_range = division_10exponential_to_k(lr_range.min(), lr_range.max(), k_fold)
        reg_range = division_10exponential_to_k(reg_range.min(), reg_range.max(), k_fold)
    print("expand gird to (%s,%s) -> (%s,%s)" % (lr_range.min(), reg_range.min(), lr_range.max(), reg_range.max()))
    return lr_range, reg_range, [(x, y) for x in lr_range for y in reg_range], best_record
