# -*- coding: utf-8 -*-
"""some functions for implementation"""

import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
import heapq
import math
from helpers import *

########################################### KNN ####################################################
def baseline_estimate(train,lamda_i,lamda_u,epochs,global_mean):
    """Estimate the user effect and item effect, build the baseline estimate model.
     Reference: "Surprise" libruary""" 
    
    #get the numbers of items and users
    num_item, num_user = train.get_shape()
    
    # initialize the user and item effect
    bu = np.zeros(num_user)
    bi = np.zeros(num_item)    
    
    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)
    
    #using Alternating Least Squares (ALS)
    for iter_ in range(epochs):
        
        #estimate the item effect
        for i,i_users in nz_item_userindices:
            dev_i = 0
            for u in i_users:
                dev_i += train[i,u] - global_mean - bu[u]

            bi[i] = dev_i / (lamda_i + len(i_users))
            
        #estimate the user effect
        for u,u_items in nz_user_itemindices:
            dev_u = 0    
            for i in u_items:
                dev_u += train[i,u] - global_mean - bi[i]

            bu[u] = dev_u / (lamda_u + len(u_items))
   
    return bu,bi

def user_based_similarity_by_pearson_baseline(train,min_support,global_mean, user_biases, item_biases, shrinkage=100):
    """Calculate the Pearson coefficient of users,build the similarity matrix
       Reference: "Surprise" libruary"""
    #get the numbers of items and users
    num_item, num_user = train.get_shape()
    train=train.toarray()
    
    # set some matrixs
    freq = np.zeros((num_user, num_user))# matrix of number of common items
    prods = np.zeros((num_user, num_user))# matrix of sum (r_ui - b_ui) * (r_vi - b_vi) for common items
    sq_diff_u = np.zeros((num_user,num_user))# matrix of sum (r_ui - b_ui)**2 for common items
    sq_diff_v = np.zeros((num_user,num_user))# matrix of sum (r_vi - b_vi)**2 for common items
    sim = np.zeros((num_user, num_user))#matrix of similatiries

    # Need this because of shrinkage. When pearson coeff is zero when support is 1
    min_support = max(2, min_support)

    # group the indices by row or column index
    nz_train, nz_item_userindices, nz_user_itemindices = build_index_groups(train)
    
    #build the similarity matrix
    for u,items_u in nz_user_itemindices:
        sim[u, u] = 1
        #we only calculate the upper triangle of the matrix,so the v starts from u+1
        for v,items_v in nz_user_itemindices[(u+1):]:  
            com_items = np.intersect1d(items_u,items_v)
            freq[u, v] = len(com_items)
            diff_u = (train[com_items,u] - (global_mean + item_biases[com_items] + user_biases[u]))
            diff_v = (train[com_items,v] - (global_mean + item_biases[com_items] + user_biases[v]))
            prods[u, v]= diff_u.T @ diff_v
            sq_diff_u[u, v] = diff_u.T @ diff_u
            sq_diff_v[u, v] = diff_v.T @ diff_v
            # if common items are smaller than min_support, we set the similarity to 0
            if freq[u, v] < min_support:
                sim[u, v] = 0
            else:
                # calculate the similarity
                sim[u, v] = prods[u, v] / (np.sqrt(sq_diff_u[u, v] * sq_diff_v[u, v]))
                # shrunk similarity
                sim[u, v] *= (freq[u, v] - 1) / (freq[u, v] - 1 + shrinkage)
            #copy for the below triangle in the similarity matrix
            sim[v, u] = sim[u, v]

    return sim

def KNN_with_user_means(train,sim_matrix,k,min_k,row,col,user_mean):
    """Predict by KNN method based on user means
       Reference: "Surprise" libruary"""
    
    #initial setting
    i=row
    u=col
    num_item, num_user = train.shape
    neighbors=[]

    #get the users who has nonzero ratings for the same movie (i)
    for v in range(num_user):
        if train[i,v]>0:
            new_neighbors=(v,sim_matrix[u, v],train[i,v])
            neighbors.append(new_neighbors)
            
    # Extract the top-K most-similar ratings
    k_neighbors = heapq.nlargest(k, neighbors, key=lambda t: t[1])

 
    # compute weighted average
    est = user_mean[u]
    sum_sim = 0
    sum_ratings = 0
    actual_k = 0
    for (nb,sim_, r) in k_neighbors:
        if sim_ > 0:
            sum_sim += sim_
            sum_ratings += (sim_ * (r - user_mean[nb]) )
            actual_k += 1
    if actual_k < min_k:
        sum_ratings = 0
    if sum_sim>0:
        est += sum_ratings / sum_sim

    # round ratings
    if est < 1:
        est = 1
    elif est > 5:
        est = 5
    else:
        est = np.round(est)       
    return est


########################################### MF ####################################################
def init_MF(train, num_features):
    """init the parameter for matrix factorization."""    
    #get the numbers of items and users
    num_item, num_user = train.get_shape()

    user_features = np.random.rand(num_features, num_user)
    item_features = np.random.rand(num_features, num_item)
    user_bias = np.random.rand(num_user)
    item_bias = np.random.rand(num_item)

    # start by item features.
    item_nnz = train.getnnz(axis=1)
    item_sum = train.sum(axis=1)

    for ind in range(num_item):
        item_features[0, ind] = item_sum[ind, 0] / item_nnz[ind]
        
    # then user features.
    user_nnz = train.getnnz(axis=0)
    user_sum = train.sum(axis=0)
    
    for ind in range(num_user):
        user_features[0, ind] = user_sum[0, ind] / user_nnz[ind]
        
    # intialize bias
    user_bias = user_features[0,:]
    item_bias = item_features[0,:]
        
    return user_features, item_features, user_bias, item_bias

def compute_error_MF(data, user_features, item_features, user_bias, item_bias, nz):
    """compute the loss (MSE) of the prediction of nonzero elements for matrix factorization."""
    mse = 0
    for row, col in nz:
        item_info = item_features[:, row]
        user_info = user_features[:, col]
        item_deviation = item_bias[row]
        user_deviation = user_bias[col]
        mse += (data[row, col] - (user_info.T.dot(item_info) + item_deviation + user_deviation)) ** 2
    return np.sqrt(1.0 * mse / len(nz))  

def matrix_factorization_SGD(train, test,gamma,lambda_user,lambda_item,num_features):
    """matrix factorization by SGD."""
    
    # define parameters
    num_epochs = 10     # number of full passes through the train set
    errors = [0]
    
    # set seed
    np.random.seed(348)

    # init matrix
    user_features, item_features, user_bias, item_bias = init_MF(train, num_features)
    
    # find the non-zero ratings indices 
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))
    
    # Global mean
    nb_nz_row, nb_nz_col = np.nonzero(train)
    global_mean = np.sum(train)/len(nb_nz_row)
    
    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):        
        # shuffle the training rating indices
        np.random.shuffle(nz_train)
        
        # decrease step size
        gamma *= 0.9
        
        for d, n in nz_train:
            # update W_d (item_features[:, d]) and Z_n (user_features[:, n])
            item_info = item_features[:, d]
            user_info = user_features[:, n]
            item_deviation = item_bias[d]
            user_deviation = user_bias[n]
            err = train[d, n] - user_info.T.dot(item_info) - item_deviation - user_deviation
    
            # calculate the gradient and update
            item_features[:, d] += gamma * (err * user_info - lambda_item * item_info)
            user_features[:, n] += gamma * (err * item_info - lambda_user * user_info)
            item_bias[d] += gamma * (err  - lambda_item * (item_deviation + user_deviation - global_mean))
            user_bias[n] += gamma * (err  - lambda_item * (item_deviation + user_deviation - global_mean))
            
        rmse = compute_error_MF(train, user_features, item_features, user_bias, item_bias, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))
        
        errors.append(rmse)

    # evaluate the test error
    rmse = compute_error_MF(test, user_features, item_features, user_bias, item_bias, nz_test)
    print("RMSE on test data: {}.".format(rmse))
    
    # output the resulting matrices
    return user_features, item_features, user_bias, item_bias     

def predict_MF(submission_row_col, user_features, item_features,user_bias,item_bias):
    """Predict the missing ratings by MF and round the results"""
    #Note the indicies saved in submission_row_col starts from row=1,col=1
    predictions = []
    for row, col in submission_row_col:
        item_info = item_features[:, row-1]
        user_info = user_features[:, col-1]
        item_deviation = item_bias[row-1]
        user_deviation = user_bias[col-1]
        value_pred = user_info.T.dot(item_info) + item_deviation + user_deviation
        
        #round results
        if value_pred < 1:
            value_pred = 1
        elif value_pred > 5:
            value_pred = 5
        else:
            value_pred = round(value_pred)
        predictions.append(value_pred) 
    return predictions

########################################### Kmeans ####################################################
def initialize_u(K,data):
    """Initialize the cluster centers (u)"""
    #initial settings
    num_item = data.shape[0]
    num_user = data.shape[1]
    item_nnz = data.getnnz(axis=1)
    item_sum = data.sum(axis=1)
    item_mean = np.zeros(num_item)

    #extract the item mean
    for ind in range(num_item):
        item_mean[ind] = item_sum[ind] / item_nnz[ind]
    
    # randomly select a subset of ratings, copy it to u 
    indices = np.random.randint(0,num_user,size=K)
    u = data[:,indices]
    u = u.toarray() # To use nonzero()
    
    # fill the zeros in centers(u) with item mean
    for i in range(K):
        for j in range(num_item):
            if j not in u[:,i].nonzero()[0]:
                u[j,i] = item_mean[j]
    return u

def update_z(u,K,data):
    """Given the centers(u),update the optimal assignments (z)"""
    #initial settings
    num_user = data.shape[1]
    num_item = data.shape[0]   
    z = np.zeros((K,num_user))
    loss = []
    
    #compute the optimal assgnments
    for n in range(num_user):
        inv = np.zeros(K)
        valid_items = np.where(data[:,n]>0)[0]  
        
        #compute the distance of user to each center
        for k in range(K):
            inv[k] = np.sum(abs(data[valid_items,n]-u[valid_items,k]))
        
        #map each user to its closest center
        znk=np.argmax(inv)
        loss.append(np.min(inv))       
        z[znk,n]=1
    return z,loss

def update_u(z,K,data,item_mean):
    """freeze the assignments and re-compute the best centers for each cluster"""
    #initial settings
    num_item = data.shape[0]
    u = np.zeros((num_item,K))
    
    #compute the best centers
    for k in range(K):
        users = np.where(z[k,:]==1)[0]
        valid_matrix=data[:,users] 
        item_nnz = np.count_nonzero(valid_matrix,axis=1)
        item_sum = valid_matrix.sum(axis=1)
        
        #compute the mean for each cluster
        for ind in range(num_item):
            if item_nnz[ind] == 0:
                u[ind,k] = item_mean[ind]
            else:
                u[ind,k] = item_sum[ind] / item_nnz[ind]            
    return u

def kmeans(data,data_, nb_times):
    """ Predict the missing ratings by K-means"""
   
    # initial settings
    loss_list = []
    max_iters = 20
    K_range = [12,14,16,18,20,22,24]     
    threshold=1e-8
    old_loss = 10000
    
    num_item = data_.shape[0]
    num_user = data_.shape[1]
    item_nnz = np.count_nonzero(data_,axis=1)
    item_sum = data_.sum(axis=1)
    item_mean = np.zeros(num_item)

    #extract item mean
    for ind in range(num_item):
        item_mean[ind] = item_sum[ind] / item_nnz[ind]    
   
    #calculate and update centers and assignments
    result = np.zeros((num_item,num_user))
    for K in set(K_range):
        
        #initialize the cluster centers
        u_old = initialize_u(K, data)
    
        # start the kmeans algorithm.
        for i in range(max_iters):
               
            # update z and u
            z,loss = update_z(u_old,K,data_)
            u = update_u(z,K,data_,item_mean)
            
            # calculate the average loss over all points
            average_loss = np.mean(loss)
            if abs(old_loss - average_loss) < 1e-6:
                break
            loss_list.append(average_loss)
            old_loss = average_loss
            print("The current iteration of k-means is: {m}, the average loss is {l}.".format(m=i, l=average_loss))
            u_old = u
                
        # Assign the prediction result
        for j in range(num_user):
            k = z[:,j].nonzero()[0][0]
            result[:,j] += u[:,k]
            
    # Take the average
    result /= (len(K_range))    
    return result

def compute_error_Kmeans(data_,result):
    """compute the loss (MSE) of the prediction of nonzero elements."""
    nz_row,nz_col = data_.nonzero()
    mse = 0
    print(len(nz_row))
    for i in range(len(nz_row)):
        mse += (data_[nz_row[i], nz_col[i]] - result[nz_row[i],nz_col][i]) ** 2
    return np.sqrt(1.0 * mse / len(nz_row))
