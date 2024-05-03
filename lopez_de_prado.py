# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 13:58:22 2023

@author: charlesr
"""

import numpy as np
import pandas as pd
import time
#from sklearn.datasets import make_classification
from sklearn.metrics import log_loss
from sklearn.model_selection._split import KFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.utils import check_random_state
from scipy.linalg import block_diag
from sklearn.metrics import mutual_info_score
import scipy.stats as stats
import multiprocessing as mp


# === Content from Machine Learning for Asset Managers ===

# =============================================================================
# def getTestData(n_features = 100, n_informative = 25, n_redundant = 25, 
#                 n_samples = 10000, random_state = 0, scale = 0):
#     
#     # Calculate the number of noise features
#     n_noise = n_features - n_informative - n_redundant
#     
#     # Generate a random dataset for a classification prolem
#     np.random.seed(random_state)
#     
#     # Use make_classification to construct informative and noise features
#     X, y = make_classification(n_samples = n_samples,
#                                n_features = n_features - n_redundant,
#                                n_informative = n_informative,
#                                n_redundant = 0,
#                                shuffle = False,
#                                random_state = random_state)
#     
#     # Add names for the informative features
#     cols = [f'I_{i}' for i in range(n_informative)]
#     
#     # Add names for the noise features
#     cols += [f'N_{i}' for i in range(n_noise)]
#     
#     # Convert results to a pandas data frame
#     X, y = pd.DataFrame(X, columns = cols), pd.Series(y)
#     
#     # Randomly choose which features the redundant ones replicate
#     rep = np.random.choice(range(n_informative), size = n_redundant)
#     
#     for j, k, in enumerate(rep):
#         
#         # Redundant feature j is informative feature k plus random noise
#         X[f'R_{j}'] = X[f'I_{k}'] + np.random.normal(size = n_samples, 
#                                                      scale = scale)
#         
#     return X, y
# =============================================================================

def featImpMDI(clf, feat_names):
    
    # Feature importance based on IS mean impurity reduction
    df = {i:tree.feature_importances_ for i, tree in enumerate(clf.estimators_)}
    
    # Convert from dictionary to data frame
    df = pd.DataFrame.from_dict(df, orient = 'index')
    
    # Name the columns
    df.columns = feat_names
    
    # Because max_features = 1
    df = df.replace(0, np.nan)
    
    # Calculate the mean and std of the samples
    imp = pd.concat({'mean':df.mean(), 'std':df.std()/np.sqrt(df.shape[0])}, 
                    axis = 1)
    
    # Rescale by dividing by mean
    imp /= imp['mean'].sum()
    
    return imp

def featImpMDA(clf, X, y, n_splits = 10):
    
    # Initialize k-folds constructor
    cv_gen = KFold(n_splits = n_splits)
    
    # Initialize pandas objects to hold raw and shuffled log_loss scores
    score_raw, score_shuff = pd.Series(), pd.DataFrame(columns = X.columns)
    
    # Generate split
    for fold, (train_idx, test_idx) in enumerate(cv_gen.split(X = X)):
        
        # Create training arrays
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        
        # Create testing arrays
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
        
        # Fit the model using the training data
        clf_fit = clf.fit(X = X_train, y = y_train)
        
        # Use testing data to predict probabilities
        probs = clf_fit.predict_proba(X_test)
        
        # Record log-loss
        score_raw.loc[fold] = -log_loss(y_test, probs, labels = clf.classes_)
        
        for col in X.columns:
            
            # Make a deep copy of X_test
            X_shuff = X_test.copy()
            
            # Shuffle the j-th column
            np.random.shuffle(X_shuff[col].values)
            
            # Predict the probabilities
            probs_shuff = clf_fit.predict_proba(X_shuff)
            
            # Calculate the score
            score_shuff.loc[fold, col] = -log_loss(y_test, probs_shuff, 
                                                   labels = clf.classes_)
    
    # Subtract the raw score from the score after the shuffle
    imp = score_shuff.sub(score_raw, axis = 0)
    
    # Normalize by dividing by the shuffled score
    imp = imp/score_shuff
    
    # Compute the mean and std
    imp = pd.concat({'mean':imp.mean(), 
                     'std':imp.std()/np.sqrt(imp.shape[0])}, axis = 1)
    
    return imp

def regFeatImpMDA(reg, X, y, n_splits = 10, p = 2):
    
    # Define penalty function
    pen_fun = lambda e: np.sum(np.abs(e)**p)
    
    # Initialize k-folds constructor
    cv_gen = KFold(n_splits = n_splits)
    
    # Initialize pandas objects to hold raw and shuffled log_loss scores
    score_raw, score_shuff = pd.Series(), pd.DataFrame(columns = X.columns)
    
    # Generate split
    for fold, (train_idx, test_idx) in enumerate(cv_gen.split(X = X)):
        
        # Create training arrays
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        
        # Create testing arrays
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
        
        # Fit the model using the training data
        reg_fit = reg.fit(X = X_train, y = y_train)
        
        # Use testing data to predict probabilities
        y_pred = reg_fit.predict(X_test)
        
        # Record log-loss
        score_raw.loc[fold] = -pen_fun(y_test - y_pred)
        
        for col in X.columns:
            
            # Make a deep copy of X_test
            X_shuff = X_test.copy()
            
            # Shuffle the j-th column
            np.random.shuffle(X_shuff[col].values)
            
            # Predict the probabilities
            y_shuff = reg_fit.predict(X_shuff)
            
            # Calculate the score
            score_shuff.loc[fold, col] = -pen_fun(y_test - y_shuff)
    
    # Subtract the raw score from the score after the shuffle
    imp = score_shuff.sub(score_raw, axis = 0)
    
    # Normalize by dividing by the shuffled score
    imp = imp/score_shuff
    
    # Compute the mean and std
    imp = pd.concat({'mean':imp.mean(), 
                     'std':imp.std()/np.sqrt(imp.shape[0])}, axis = 1)
    
    return imp
        

# =============================================================================
# X, y = getTestData(40, 5, 30, 10000, 2)
# 
# clf = DecisionTreeClassifier(criterion = 'entropy', max_features = 1, 
#                              class_weight = 'balanced')
# 
# clf = BaggingClassifier(estimator = clf, n_estimators = 1000,
#                         max_features = 1.0, max_samples = 1.0, 
#                         oob_score = False)
#  
# imp = featImpMDA(clf, X, y, 5)  
# =============================================================================

def groupMeanStd(df, clusters):
    
    # Initialize data frame for output
    out = pd.DataFrame(columns = ['mean', 'std'])
    
    # Loop over clusters
    for clst, col in clusters.items():
        
        # Take the sum of the values for each cluster
        temp = df[col].sum(axis = 1)
        
        # Compute the mean value
        out.loc[f'C_{clst}', 'mean'] = temp.mean()
        
        # Compute the standard deviation 
        out.loc[f'C_{clst}', 'std'] = temp.std()/np.sqrt(temp.shape[0])
        
    return out

def featImpMDI_Clustered(clf_fit, feat_names, clusters):
    
    # Feature importance based on IS mean impurity reduction
    df = {i:tree.feature_importances_ for i, tree in enumerate(clf_fit.estimators_)}
    
    # Convert dictionary to data frame
    df = pd.DataFrame.from_dict(df, orient = 'index')
    
    # Rename columns
    df.columns = feat_names
    
    # Because max_features = 1
    df = df.replace(0, np.nan)
    
    # Get impurity of each cluster
    imp = groupMeanStd(df, clusters)
    
    # Divide by sum to normalize
    imp /= imp['mean'].sum()
    
    return imp

def featImpMDA_Clustered(clf, X, y, clusters, n_splits = 10):
    
    # Initialize k-folds constructor
    cv_gen = KFold(n_splits = n_splits)
    
    # Initialize pandas objects to hold raw and shuffled log_loss scores
    score_raw, score_shuff = pd.Series(), pd.DataFrame(columns = clusters.keys())
    
    # Generate splits
    for fold, (train_idx, test_idx) in enumerate(cv_gen.split(X = X)):
        
        # Create training arrays
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        
        # Create testing arrays
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
        
        # Fit the model using training data
        clf_fit = clf.fit(X = X_train, y = y_train)
        
        # Use the fitted model to predict probabilities
        probs = clf_fit.predict_proba(X_test)
        
        # Record log-loss
        score_raw.loc[fold] = -log_loss(y_test, probs, labels = clf.classes_)
        
        # Loop over clusters
        for clst in clusters:
            
            # Make a deeop copy of X_test
            X_shuff = X_test.copy()
            
            # For each column in clst
            for col in clusters[clst]:
                
                # Shuffle col
                np.random.shuffle(X_shuff[col].values)
            
            # Predict the probabilities with shuffled results
            probs_shuff = clf_fit.predict_proba(X_shuff)
            
            # Calcualte the score
            score_shuff.loc[fold, clst] = -log_loss(y_test, probs_shuff, 
                                                    labels = clf.classes_)
    
    # Subtract the raw score from the score after the shuffle
    imp = score_shuff.sub(score_raw, axis = 0)
    
    # Normalize by the shuffled scores
    imp = imp/score_shuff
    
    # Calculate the mean and std
    imp = pd.concat({'mean':imp.mean(), 
                     'std':imp.std()/np.sqrt(imp.shape[0])}, axis = 1)
    
    # Change the index name
    imp.index = [f'C_{i}' for i in imp.index]
        
    return imp


def regFeatImpMDA_Clustered(reg, X, y, clusters, n_splits = 10, p = 2):
    
    # Define penalty function
    pen_fun = lambda e: np.sum(np.abs(e)**p)
    
    # Initialize k-folds constructor
    cv_gen = KFold(n_splits = n_splits)
    
    # Initialize pandas objects to hold raw and shuffled log_loss scores
    score_raw, score_shuff = pd.Series(), pd.DataFrame(columns = clusters.keys())
    
    # Generate splits
    for fold, (train_idx, test_idx) in enumerate(cv_gen.split(X = X)):
        
        # Create training arrays
        X_train, y_train = X.iloc[train_idx, :], y.iloc[train_idx]
        
        # Create testing arrays
        X_test, y_test = X.iloc[test_idx, :], y.iloc[test_idx]
        
        # Fit the model using training data
        reg_fit = reg.fit(X = X_train, y = y_train)
        
        # Use the fitted model to predict probabilities
        y_pred = reg_fit.predict(X_test)
        
        # Record log-loss
        score_raw.loc[fold] = -pen_fun(y_test - y_pred)
        
        # Loop over clusters
        for clst in clusters:
            
            # Make a deeop copy of X_test
            X_shuff = X_test.copy()
            
            # For each column in clst
            for col in clusters[clst]:
                
                # Shuffle col
                np.random.shuffle(X_shuff[col].values)
            
            # Predict the probabilities with shuffled results
            y_shuff = reg_fit.predict(X_shuff)
            
            # Calcualte the score
            score_shuff.loc[fold, clst] = -pen_fun(y_test - y_shuff)
    
    # Subtract the raw score from the score after the shuffle
    imp = score_shuff.sub(score_raw, axis = 0)
    
    # Normalize by the shuffled scores
    imp = imp/score_shuff
    
    # Calculate the mean and std
    imp = pd.concat({'mean':imp.mean(), 
                     'std':imp.std()/np.sqrt(imp.shape[0])}, axis = 1)
    
    # Change the index name
    imp.index = [f'C_{i}' for i in imp.index]
        
    return imp

# === Functions from "Machine Learning for Asset Managers" by Lopez de Prado pages 52-59 ===
def clusterKMeansBase(corr0, maxNumClusters = 10, n_init = 10):
    
    # Calculate distance matrix
    x = np.sqrt(0.5 * (1 - corr0.fillna(0)).clip(lower = 0, upper = 2))
    
    # Initialize pandas series to hold silhouette scores
    silh = pd.Series()
    
    # Loop over possible number of clusters
    for i in range(2, maxNumClusters + 1):
        
        # Itialize k-means
        kmeans_ = KMeans(n_clusters = i, n_init = n_init)
        
        # Fit results
        kmeans_ = kmeans_.fit(x)
        
        if len(np.unique(kmeans_.labels_)) > 1:
            
            # Get silhouette scores
            silh_ = silhouette_samples(x, kmeans_.labels_)
            
            # Record results
            stat = (silh_.mean()/silh_.std(), silh.mean()/silh.std())
            
            # If improvement...
            if np.isnan(stat[1])|(stat[0] > stat[1]):
                
                # ... record results
                silh, kmeans = silh_, kmeans_
                
        else:
            
            continue
    
    # Get the index values
    idx_new = np.argsort(kmeans.labels_)
    
    # Rearrange correlation matrix
    corr1 = corr0.iloc[idx_new, idx_new]
    
    # Create dictionary to record the clusters
    clust_dict = {clst:corr0.columns[np.where(kmeans.labels_ == clst)[0]].tolist() for clst in np.unique(kmeans.labels_)}
    
    # Change index of silhouette scores
    silh = pd.Series(silh, index = x.index)
    
    return corr1, clust_dict, silh

    
def makeNewOutputs(corr0, clusters, clusters2):
    
    # Initialize clusters dictionary
    clusters_new = {}
    
    # Loop over the original cluster keys
    for clst in clusters:
        
        # Save the results; key based on order in clusters
        clusters_new[len(clusters_new.keys())] = list(clusters[clst])
    
    # Loop over the new cluster keys
    for clst in clusters2:
        
        # Save results; key based on order in clusters2 + number in original clusters
        clusters_new[len(clusters_new.keys())] = list(clusters2[clst])
    
    # Get the new indices 
    idx_new = [idx for clst in clusters_new for idx in clusters_new[clst]]
    
    # Reorder observations based on cluster order
    corr_new = corr0.loc[idx_new, idx_new]

    # Calculate distance matrix
    x = np.sqrt(0.5 * (1 - corr0.fillna(0)).clip(lower = 0, upper = 2))
    
    # Initialize labels
    kmeans_labels = np.zeros(x.shape[0])
    
    # Loop over new cluster keys
    for clst in clusters_new:
        
        # Get the indacies in cluster clst
        clst_idxs = [x.index.get_loc(k) for k in clusters_new[clst]]
        
        # Save these
        kmeans_labels[clst_idxs] = clst
    
    # Record new Silhouette scores
    silh_new = pd.Series(silhouette_samples(x, kmeans_labels), index = x.index)
    
    return corr_new, clusters_new, silh_new

def clusterKMeansTop(corr0, maxNumClusters = None, n_init = 10):
    
    # If the maximum number of clusters is None, then set it equal to the number of columns minus 1
    if maxNumClusters is None: maxNumClusters = corr0.shape[0] - 1
    
    # Perform clustering
    corr1, clusters, silh = clusterKMeansBase(corr0, 
                                            maxNumClusters = 
                                            np.min([maxNumClusters, corr0.shape[0] - 1]), 
                                            n_init = n_init)
    
    # Calculate t-stat for each cluster
    t_stats = {clst:np.mean(silh[clusters[clst]])/np.std(silh[clusters[clst]]) for clst in clusters}
    
    # Compute the mean t-stat
    t_stats_mean = np.mean(list(t_stats.values()))
    
    # Get list of clusters to redo
    redo = [clst for clst in t_stats if t_stats[clst] < t_stats_mean]
    
    # If one terminate algorithm 
    if len(redo) <= 1:
        
        return corr1, clusters, silh
    
    # Otherwise
    else:
        
        # Get the keys for the observations to redo
        redo_keys = [key for clst in redo for key in clusters[clst]]
        
        # Subset the observation matrix
        corr_temp = corr0.loc[redo_keys, redo_keys]
        
        # Calculate the mean t-stat
        t_stats_mean = np.mean([t_stats[clst] for clst in redo])
        
        # Perform new clustering
        corr2, clusters2, silh2 = clusterKMeansTop(corr_temp, 
                                                 maxNumClusters = np.min([maxNumClusters, corr_temp.shape[0] - 1]),
                                                 n_init = n_init)
        
        # Make new outputs if necessary
        corr_new, clusters_new, silh_new = makeNewOutputs(corr0, 
                                                          {clst:clusters[clst] for clst in clusters if clst not in redo}, 
                                                          clusters2)
        
        # Calculate t-stats of newly generated clusters
        new_t_stats_mean = np.mean([np.mean(silh_new[clusters_new[clst]])/np.std(silh_new[clusters_new[clst]]) for clst in clusters_new])
        
        if new_t_stats_mean <= t_stats_mean:
            
            return corr1, clusters, silh 
        
        else:
            
            return corr_new, clusters_new, silh_new 
        

def getCovSub(nObs, nCols, noise, random_state = None):
    
    # Sub correlation matrix
    rnd = check_random_state(random_state)
    
    if nCols == 1: return np.ones(shape = (1, 1))
    
    # Geneate random normals
    x = rnd.normal(size = (nObs, 1))
    
    # Repeat nCols times and stack horizontally
    X = np.repeat(x, nCols, axis = 1)
    
    # Add random noise 
    X += rnd.normal(scale = noise, size = X.shape)
    
    # Calculate the covariance matrix
    cov = np.cov(X, rowvar = False)
    
    return cov

def getRndBlockCov(nCols, nBlocks, minBlockSize = 1, noise = 1.0, random_state = None):
    
    # Generate a block random correlation matrix
    rnd = check_random_state(random_state)
    
    # Generate ranom numebers without replacement
    parts = rnd.choice(range(1, nCols - (minBlockSize - 1) * nBlocks), 
                       nBlocks - 1, replace = False)
    
    # Sort values
    parts.sort()
    
    # Add N' as the last value
    parts = np.append(parts, nCols - (minBlockSize - 1) * nBlocks)
    
    # Take the difference, subtract 1 and add M to get the number in each block
    parts = np.append(parts[0], np.diff(parts)) - 1 + minBlockSize
    
    # Initialize covariance matrix
    cov = None
    
    # Loop over each block
    for nCols_ in parts:
        
        # Calculate the number of observations for each block
        nObs_ = int(np.max([nCols_ * (nCols_ + 1)/2, 100]))
        
        # Get the covariance matrix for black
        cov_ = getCovSub(nObs_, nCols_, noise, random_state = rnd)
        
        # Construct block covariance matrix
        cov = cov_.copy() if cov is None else block_diag(cov, cov_)
        
    return cov

def randomBlockCorr(nCols, nBlocks, minBlockSize = 1, random_state = None):
    
    # Form block correlation
    rnd = check_random_state(random_state)
    
    # Calculate signal covariance matrix
    cov = getRndBlockCov(nCols, nBlocks, minBlockSize = minBlockSize, 
                          noise = 0.5, random_state = rnd)
    
    # Create noise covariance matrix
    cov_noise = getRndBlockCov(nCols, 1, minBlockSize = minBlockSize, 
                               noise = 1.0, random_state = rnd)
    
    # Add noise covariance matrix to signal covariance matrix
    cov += cov_noise
    
    # Create diagonal matrix with entries 1 over std
    sd_inv = np.diag(1/np.sqrt(np.diag(cov)))
    
    # Convert covariance matrix to correlation matrix
    corr = sd_inv  @ cov @ sd_inv
    
    # Clip at -1 and 1 for numerical stability
    corr = pd.DataFrame(corr).clip(-1, 1)
    
    return corr

# From 'Machine Learning for Asset Managers' by Lopez de Prado page 44-46 (for marginal entropy)
def opt_bins(N, corr = None):
    
    # Hacine-Gharbi optimimal number of bins...
    if (corr is None)|(corr >= 1 - 1e-9)|(corr != corr):
        
        # ... for entropy results
        z = (8 + 324 * N + 12 * np.sqrt(36 * N + 729 * N**2))**(1/3)
        b = z/6 + 2/(3 * z) + 1/3
        
    else:
        
        # ... for mutual information results
        b = 1/np.sqrt(2) * np.sqrt(1 + np.sqrt(1 + 24 * N/(1 - corr**2)))
    
    # Round to integer
    return int(b + 0.5)

def mutual_info(x, y, normalize = False):
    
    # Find the optimal number of bins
    bins = opt_bins(x.shape[0], corr = np.corrcoef(x, y)[0, 1])
    
    # Calculate a histogram of results
    histogram = np.histogram2d(x, y, bins)[0]
    
    # Use histogram to calculate mutual information
    I = mutual_info_score(None, None, contingency = histogram)
    
    if normalize == True:
        
        # Calculate entropy of X
        H_X = stats.entropy(np.histogram(x, bins)[0])
        
        # Calculate entropy of Y
        H_Y = stats.entropy(np.histogram(y, bins)[0])
        
        # Normalize I using the minmum of these two
        I /= np.min([H_X, H_Y])
        
    return I

def calc_mut_info_mat(df):
    
    # Record the list of variables
    var_list = df.columns
    
    # Create data frame of mutual information
    mut_info = pd.DataFrame(index = var_list, columns = var_list)
   
    # Loop over the elements of var_list
    for i, idx in enumerate(var_list):
        
        # Loop over the lements that haven't occured
        for col in var_list[i:]:
            
            # Drop missing values
            temp = df[[idx, col]].dropna(axis = 0)
            
            if idx == col:
                
                # If on the diagonal, make 1
                mut_info.loc[idx, col] = 1
                
            else:
                
                # If not on diagonal calculate mutual information
                I = mutual_info(temp[idx].values, temp[col].values, 
                                      normalize = True)
                
                # Fill positions with this value
                mut_info.loc[idx, col], mut_info.loc[col, idx] = I, I
                
    return mut_info
 



# === Content from Advances in Financila Machine Learning ===


 
def linear_parts(atoms, threads):
    
    # Partition of atoms with a single loop
    parts = np.linspace(0, atoms, min(atoms, threads) + 1)
    parts = np.ceil(parts).astype(int)
    
    return parts

def nested_parts(atoms, threads, upper_triange = False):
    
    # partition of atoms with an inner loop
    parts, threads_ = [0], min(atoms, threads)
    
    for _ in range(threads_):
        
        part = 1 + 4 * (parts[-1]**2 + parts[-1] + atoms * (atoms + 1)/threads_)
        part = (-1 + np.sqrt(part))/2
        parts.append(part)
        
    parts = np.round(parts).astype(int)
    
    # If true make the first rows are the heaviest
    if upper_triange: 
    
        parts = np.cumsum(np.diff(parts)[::-1])
        parts = np.append(np.array([0]), parts)
        
    return parts

def expand_call(kargs):
    
    # Get the function arguement
    func = kargs['func']
    
    # Delete it from the fictionary
    del kargs['func']
    
    # Evaluate function with other arguments
    out = func(**kargs)
    
    return out

def process_jobs_single_core(jobs):
    
    print('We have begun processing!\n')
    
    # Run jobs sequentially for debugging
    out = [expand_call(job) for job in jobs]
        
    return out

def report_progress(job_num, num_jobs, start_time):
    
    # Report progress as asynch jobs are completed
    message_stats = [job_num/num_jobs, (time.perf_counter() - start_time)/60]
    message_stats.append(message_stats[1] * (1/message_stats[0] - 1))
    
    time_stamp = time.strftime("%m-%d %H:%M:%S")
    
    message = f'{time_stamp}:'
    message += f'{100 * message_stats[0]: .2f}% complete. '
    message += f'This job took {message_stats[1]:.2f} minutes. ' 
    message += f'About {message_stats[2]:.2f} minutes left.' 
    message += '\n'
    
    print(message)
    
    if job_num == num_jobs:
        
        print('Processing is complete!\n')
        

def process_jobs(jobs,  num_threads = 6):
    
    print('We have begun multiprocessing!\n')
    
    # Initialize pool and specify the number of threads
    pool = mp.Pool(processes = num_threads)
    
    # Run imap_unordered; we need index in function to keep track of order
    outputs = pool.imap_unordered(expand_call, jobs)
    
    # Initialize list to contain output
    out = []
    
    # Record time
    start_time = time.perf_counter()
    
    # Process asynchronous output, report progress
    for job_num, out_ in enumerate(outputs, 1):

        # Append results
        out.append(out_)
        
        # Report progress
        report_progress(job_num, len(jobs), start_time)
        
    pool.close()
    pool.join()
    
    return out

# Wrapper to vectorize; hard to pickle np.vectorize because it runs in C
# See https://stackoverflow.com/questions/78307097/multiprocessing-pool-imap-unordered-cant-pickle-function/78307726?noredirect=1#comment138058459_78307726
class vectorize_wrapper:
    
    def __init__(self, pyfunc):
        
        self.__name__ = 'wrapped_' + pyfunc.__name__ 
        
        self.func = np.vectorize(pyfunc)


    def __call__(self, index, *args, **kwargs):
        
        # Convert index to pandas data frame
        index_df = pd.DataFrame(index, columns = ['index'])
            
        # Convert function output to data frame
        out_df = pd.DataFrame(self.func(*args, **kwargs))
           
        if out_df.shape[0] == index_df.shape[0]:
            
            return pd.concat([index_df, out_df], axis = 1)
        
        elif out_df.shape[1] == index_df.shape[0]:

            return pd.concat([index_df, out_df.T], axis = 1)
        
        else:
            
            raise ValueError("The dimensions are inconsistent!")
            
    def __setstate__(self, state):
        
        self.func = np.vectorize(state)

    def __getstate__(self):
        
        return self.func.pyfunc
  


def run_queued_multiprocessing(func, index, params_dict = {}, 
                               num_threads = 24, mp_batches = 1, 
                               linear_molecules = False, prep_func = True, 
                               **kwargs):
    """
    Parallelize jobs, returns a data frame or series.

    Parameters
    ----------
    func : function
        Function to be parallelized. Output must be a pandas data frame if 
        prep_func =  False.
    index : list, numpy array, pandas index, or pandas series. Used to keep 
        track of returned observations. If prep_func = False, then only used for
        number of observations
    params_dict: dictionary, optional
        Contains a dictionary of the variables to input into func. The keys are
        the argument names and the values are pandas series of the corresponding
        values. Default is {}
    num_threads : int, optional
        The number of threads that will be used in parallel (one processor per thread). 
        The default is 24.
    mp_batches : TYPE, optional
        Number of parallel batches (jobs per core). The default is 1.
    linear_molecules : boolean, optional
        Whether partitions will be linear or double-nested. The default is False.
    prep_func: boolean, optional
        Whether to vectorize function and make the first input the index. 
        Functions vectorized using np.vectorize are not pickleable so care must
        be taken to prep the functions if done manually. Furthermore, 
        mp.imap_unordered does not preserve order. As a result, the function 
        must be constructed so that inputs and outputs match.
    kwargs:
        Additional arguments of function that do not need to be vectorized.

    Returns
    -------
    Pandas data frame of sorted outputs

    """
        
    if prep_func:
        
        # Modify function
        new_func = vectorize_wrapper(func)
    
        # Add index to the parameters
        params_dict['index'] = index
    
    # Get observations
    num_obs = len(index)
    
    # Define how we're doing to break up the taks
    if linear_molecules: 
        
        parts = linear_parts(num_obs, num_threads * mp_batches)
        
    else:
        
        parts = nested_parts(num_obs, num_threads * mp_batches)
    
    # Initialize list to hold jobs
    jobs = []
    
    # Creaete jobs
    for i in range(1, len(parts)):
        
        job = {key:params_dict[key][parts[i - 1]:parts[i]] for key in params_dict}
        
        if prep_func:
            
            job.update({'func':new_func, **kwargs})
            
        else:
            
            job.update({'func':func, **kwargs})
            
        jobs.append(job)
        
    # If number of threads is one...   
    if num_threads == 1: 
        
        # ... run simply using list comprehension
        out = process_jobs_single_core(jobs)
    
    # Otherwise...
    else:
        
        # ... use multiprocessing module
        out = process_jobs(jobs, num_threads = num_threads)
        
    
    # Concatinate results in list
    result_df = pd.concat(out, axis = 0)
    
    if prep_func:
        
        # Set index as the index and drop as column
        result_df = result_df.set_index('index', drop = True)

    # Sort by the index
    result_df = result_df.sort_index()   
    
    return result_df      
