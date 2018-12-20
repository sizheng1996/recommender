# Machine Learning Project 2: Recommender System


This project aims at creating a recommander system and competing with other teams on crowdai. This file contains the preprocessing of data, multi-splitting data set, the implementation of three Collaborative Filtering methods, and the creation of submission to crowdai.org.


|Author|Jiahua Wu|Siqi Zheng|
|---|---|---
|E-mail|jiahua.wu@epfl.ch|siqi.zheng@epfl.ch

|crowdAI username|submission ID|
|---|---
|E-mail|jiahua.wu@epfl.ch
----

crowdAI username but additionally also the crowdAI submission ID number which corresponds to your code submission
## List of files
- run.ipynb
- implementations. py
- helpers. py
- plots. py
- dataset_train.csv
- dataset_submission.csv
## Note
It will take no more than 10 mins for the `run.ipynb` to run. Please be patient during the process.
## Get started
These instructions will help you make`run.ipynb` running on your local machine and obtain the same result as ours.
## Data loading and Data Spliting
#### Data Loading
The training set is loaded from ``dataset_train.csv`` and is transformed into ``scipy.sparse.lil_matrix`` type to facilitate later operation. Also, we create its copy `ratings_` of type `numpy.array`.
For submission, we load the submission sample and transform the first column which contains place information of the missing ratings to a list of tuples `submission_row_col`. The original text of the place information is saved in `submission_pos`.
#### Data Spliting
The objective of function `split_data` is to split the training set into "train set" and "test set" so that we can do a local test to estimate the performance of the predictor.The spliting ratio could be changed by tuning parameter `p_test` Moreover, a function `multi_data_split`is provided for have multi training and testing data set for selecting parameters.

## Data preprocessing
Data preprocessing includes three parts: 1.Extract mean; 2.Baseline estimate; 3.Similarity measure with Pearson coefficient. Part 1 will output user mean and global mean. Part 2 will calculate the item effect and user effect used for part 3. Three parameters `lamda_i`, `lamda_u`, `epochs`could be tuned. Part 3 will give the similarity matrix.`min_support`, `shrinkage`two parameters could be tuned.

## Three Collaborative Filtering methods
- K-Nearest Neighbors with means  
The users can change the parameters `k `and `min_k`. `k `is the number of nearest neighbors. The default value of `k` is 200. `min_k` is the minimum number of support neighbors. The default value of `min_k `is 1. 
- Matrix Factorization  
The users can change the parameters `lambda_user `and `lambda_item`. `lambda_user` is the regularization parameter of user. The default value of `lambda_user` is 1e-3.4. `lambda_item` is the regularization parameter of item. The default value of `lambda_item` is 100.
- K-means  
If users want to change the parameter( for example, the number of cluster `K`), they need to go to the function `kmeans` in the file `implementations.py` and change the `K_range`.

## Submission
By default, solution of the KNN method will be automatically written in `"prediction_recommender.csv"`. The users can output the other results (pred_MF,pred_Kmeans) by changing the input parameter `pred_KNN` in the function`create_csv_submission`.

## Code Example
```
def knn_demo(train,test_row_col,nonzero_test,sim_matrix):
    """Prediction by KNN method, return the rmse for train or test data"""
    pred=[]    
    for row,col in test_row_col:
        est = KNN_with_user_means(train,sim_matrix,k,min_k,row,col,user_mean[0].T)
        pred.append(est)  
  
    error = nonzero_test-np.array(pred)
    rmse = np.sqrt( error.dot(error.T)/error.shape[1] )
    return rmse
```

## TODO

- Data preprocessing (Reduce dimensionality)
- New CF methods (Restricted Boltzmann Machine, Decision Trees)

## Reference
http://surpriselib.com

