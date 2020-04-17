import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

"""
Code adapted from 
Shi Q, Zhang C, Guo W, Zeng T, Lu L, Jiang Z, et al. 
Local network component analysis for quantifying transcription factor activities. Methods. 2017;124: 25–35 
https://www.sciencedirect.com/science/article/pii/S1046202317300506?via=ihub

"""

def local_network_component_analysis(X, K, NETWORK, lam_1, iter_num, TF_num):
    """
    Learn transcription factor activity through local similarities using KNN algorithm
    :param X: pd.dataframe [G x S]. Expression matrix - genes x samples
    :param K: int, K number of neighbors for KNN
    :param NETWORK: pd.dataframe [G x TFs]. Prior matrix - genes x TFs
    :param lam_1: int, default parameter
    :param iter_num: int, default iteration
    :param TF_num: int, number of TFs
    :return: A_vec, Y
    """


    """
    X is sample*gene; K is neighbor number for selecting window
    lam_1 is parameter
    A is the set of TF-gene network of each sample
    Y is the activation score of TFs
    """

    # number of samples
    n = X.shape[1]

    # capturing the sample window by KNN
    window_id_model = NearestNeighbors(n_neighbors=K)
    window_id_model.fit(X.T)

    # matrix of indices of neighbors
    window_id = window_id_model.kneighbors(X.T)[1]

    # capturing index S_i
    S_vec = []
    for i in range(n):
        S_i = np.zeros([n, K])
        for temp in range(K):
            S_i[window_id[i,temp]][temp] = 1
        S_vec.append(S_i)

    # capturing the weight w_i(j)
    W_vec = []
    for i in range(n):
        W_i = np.zeros([n, n])
        for temp in range(K):
            W_i[window_id[i, temp]][window_id[i, temp]] = np.sqrt(1 / (K * n))
        W_vec.append(W_i)

    # Running the FastNCA for each sample neighborhood
    for j in range(iter_num):
        A_vec = []
        Y_vec = []
        temp_window_id = []
        temp_W_vec = []
        temp_S_vec = []

        for i in range(len(S_vec)):
            if np.count_nonzero(W_vec[i] @ (S_vec[i])) > TF_num:

                A_tmp, Y_tmp = fast_network_component_analysis(X @ W_vec[i] @ S_vec[i], NETWORK)

                for k in range(A_tmp.shape[1]):
                    sum_temp = A_tmp[:, k].sum()
                    A_tmp[:, k] = A_tmp[:, k] / sum_temp
                    Y_tmp[k, :] = Y_tmp[k, :] * sum_temp

                A_vec.append(A_tmp)
                Y_vec.append(Y_tmp)

                temp_W_vec.append(W_vec[i])
                temp_S_vec.append(S_vec[i])

                temp_window_id.append(window_id[i][:])

        # capture the global TF activation by aligning the local TF activation
        W_vec = temp_W_vec
        S_vec = temp_S_vec
        window_id = temp_window_id

        Y = align_TF_activation(Y_vec, S_vec)

        # compute the error
        err = []
        for i in range(len(S_vec)):
            temp_err = np.diag((X @ S_vec[i] - A_vec[i] @ Y @ S_vec[i]).T @ (X@S_vec[i]-A_vec[i]@ Y@ S_vec[i]))
            normal_err = np.diag((X @ S_vec[i]).T @ (X@ S_vec[i]))
            err=np.concatenate((err, temp_err/normal_err))

        # optimize the weight of each sample in each neighbor
        W_vec = optimize_weight(err, window_id, lam_1, K, n)

    return A_vec, Y


def optimize_weight(err, window_id, lam_1, k, n):
    """

    :param err:
    :param window_id:
    :param lam_1:
    :param k:
    :param n:
    :return:
    """

    # optimizing the weight of samples

    err_n = err / err.sum()
    err_s= np.sort(err_n)
    err_index = np.argsort(err_n)

    p = np.inf
    W = np.zeros(err_n.shape[0], dtype=float)

    for i in reversed(range(err_n.shape[0])):
        o = (2 * lam_1 + err_s[0:i].sum()) / i - err_s[i]

        if o >= 0:
            p = i
            break

    o = (2 * lam_1 + err_s[0:p].sum()) / p
    W[0:p+1] = (o - err_s[0:p+1]) / (2 * lam_1)
    W[p+1: err_n.shape[0]] = 0

    W_1 = -100 * np.zeros(W.shape[0])
    W_1[err_index] = W
    W = W_1

    # constructing the W_vec
    W_vec = []
    for i in range(len(window_id)):
        #range(window_id.kneighbors(X.T)[1].shape[0]):
        W_i = np.zeros([n, n])
        for temp in range(k):
            W_i[window_id[i][temp]][window_id[i][temp]] = np.sqrt(W[((i + 1) - 1) * k + temp])
            # W_i[window_id.kneighbors(X.T)[1][i][temp]][window_id.kneighbors(X.T)[1][i][temp]] = np.sqrt(W[((i+1) - 1) * k+temp])
        W_vec.append(W_i)

    return W_vec


def fast_network_component_analysis(X, A):
    """
    FastNCA adapted from
    Chang C, Ding Z, Hung YS, Fung PCW. Fast network component analysis (FastNCA)
    for gene regulatory network reconstruction from microarray data. Bioinformatics. 2008;24: 1349–1358.
    :param:
    X: expression data
    A: connectivity matrix, NETWORK
    Model: X = AS + G
    X: the expression data, A: connectivity matrix, S: TFA matrix, G: noise
    A is sparse and satisfies the NCA criteria
    Problem: estimate A and S from X, with a known structure of A
    :return:
    Ae: estimate of the connectivity matrix
    Se: estimate of the TFA matrix
    """
    M = A.shape[1]
    A = np.array(A)

    U, S, V = np.linalg.svd(X)
    # M = min(M, S.shape[0])
    XM = U[:, 0:M] @ np.diag(S[0:M]) @ V[0:M, :]
    U = U[:, 0:M]  # M-dimensional signal subspace of the data matrix

    Ae = np.array(A).astype(float)
    for l in range(M):
        # U0 = U[np.where(NETWORK.iloc[:, l] == 0)[0], :]
        # U0 = U[np.where(NETWORK[:][l] == 0)[0], :]
        U0 = U[np.where(A[:][l] == 0)[0], :]

        if U0.shape[0] < M:
            UU, SS, VV = np.linalg.svd(U0)
            t = VV[-1, :].T
        else:
            UU, SS, VV = np.linalg.svd(U0.T)
            t = UU[:, -1]

        a = U @ t
        a = np.multiply(a, A[:, l] != 0)
        Ae[:, l] = a / np.abs(a).sum() * np.count_nonzero(A[:, l])

    # get the estimate of the TFA matrix
    Se = np.linalg.pinv(Ae) @ XM

    return Ae, Se


def align_TF_activation(Y_vec, S_vec):
    """

    :param Y_vec:
    :param S_vec:
    :return:
    """
    # obtaining matrix for solving the Y

    res_1 = Y_vec[0] @ S_vec[0].T
    res_2 = S_vec[0] @ S_vec[0].T
    for i in range(1, len(S_vec)):
        res_1 = res_1 + Y_vec[i] @ S_vec[i].T
        res_2 = res_2 + S_vec[i] @ S_vec[i].T

    Y = res_1@np.linalg.inv(res_2)

    return Y

def clean_expression(X):
    """
    If X is not full rank clean X to be full rank
    :param X: raw input expression
    :return: X full rank
    """

    # remove duplicate samples (columns) from the expression matrix
    X_dup = X.T.drop_duplicates(keep = "first")
    X_T = X_dup.T
    return(X_T)


def clean_network(NETWORK):
    """
    If NETWORK contains columns that sum to zero, clean data
    :param NETWORK: input network
    :return: network containing non-zero columns
    """
    # remove all columns that add to zero from NETWORK
    NETWORK2 = NETWORK.loc[:, (NETWORK != 0).any(axis=0)]
    return(NETWORK2)


def main():
    # Read files in
    X = pd.read_csv("../Data/input_expression.csv", sep=',', header=None)
    NETWORK = pd.read_csv("../Data/input_network.csv", sep=',', header=None)

    # check that X is full rank
    if np.linalg.matrix_rank(X) < X.shape[1]:
        print("Expression data is not full rank, fixing matrix...")
        X = clean_expression(X)
    else:
        print("Expression data is full rank")

    # check that prior columns do not sum to zero
    if (NETWORK.sum() == 0).sum() > 0:
        print("Cleaning prior")
        clean_network(NETWORK)
    else:
        print("Prior is full rank")

    # Parameters
    K = 50  # the parameter of k of KNN algorithm. must be > number of samples
    lam_1 = 1  # default parameter
    iter_num = 1  # iteration number
    TF_num = NETWORK.shape[1]  # TF number

    A_vec, Y = local_network_component_analysis(X, K, NETWORK, lam_1, iter_num, TF_num)

    print(A_vec)
    print(Y)
    np.savetxt('TF_activities.csv', Y, delimiter=',')

if __name__ == "__main__":
    main()