# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:02:37 2024

@author: charlesr
"""
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss


# Define function to get Ledoit mean estimate
def ledoit_mean(X, beta):
    
    # Calculate sample mean
    m_sample = X.mean()
    
    # Calculate shrinkage target
    m_target = np.mean(m_sample)
    
    # Calculate shrunken mean estimate
    m_shrunk = beta * m_sample + (1 - beta) * m_target
    
    return m_shrunk 


# See 'Machine Learning for Asset Managers' by Lopez de Prado page 24-30 for details on the following two functions

# Follows code at https://github.com/GGiecold/pyRMT/blob/main/pyRMT.py very closely
def marcenko_pastur(T, N):
    
    # Calculate q
    q = N/T
    
    # Get the minimum lamda
    lam_min = (1 - np.sqrt(q))**2
    
    # Get the maximum lambda
    lam_max = (1 + np.sqrt(q))**2

    return lam_max, lam_min 


# Create function to clip covariance matrix
def clip_cov(S, threshold, epsilon = 1e-8):

    # Get the standard deviations
    D = np.diag(np.sqrt(np.diag(S)))
    
    # Calculate the correlation matrix
    corr = (1/D) @ S @ (1/D)
    
    # Floor and cap to mitigate possible rounding error
    corr = np.clip(corr, -1.0, 1.0)
    
    # Fill diagonal with 1
    np.fill_diagonal(corr, 1.0)
                                 
    # Get the eigenvalues and vectors for corr
    eigvals, eigvecs = np.linalg.eigh(corr)
    
    # Make sure eigenvalues are positive
    eigvals[eigvals < epsilon] = epsilon
    
    # Clip eigen values
    eigvals[eigvals < threshold] = np.mean(eigvals[eigvals < threshold])
    
    # Get the clipped correlation
    corr_clip = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Get diagonal with entries 1 over the square root of corr_clip diagional
    E = np.diag(np.sqrt(np.diag(corr_clip)))
    
    # Make sure still correlation matrix
    corr_clip = (1/E) @ corr_clip @ (1/E)
    
    # Floor and cap to mantain numerical stability
    corr_clip = np.clip(corr_clip, -1.0, 1.0)
    
    # Compute the clipped covariance
    S_clip = D @ corr_clip @ D
    
    return S_clip
    

# Follows code at https://github.com/GGiecold/pyRMT/blob/main/pyRMT.py very closely
def get_clip_cov(X):
    
    # Get the sample coveriance matrix
    S_sample = np.cov(X, rowvar = False)
    
    # Get dimension
    T, N = X.shape
       
    # Get the lambda min and max
    lam_max, _ = marcenko_pastur(T, N)
    
    # Calculate clipped covariance matrix
    S_clip = clip_cov(S_sample, lam_max)
    
    return S_clip


# Create function to get covariance target
def get_cov_target(S):
    
    # Calculate mean of diagonal
    diag_mean = np.mean(np.diag(S))
      
    # Create matrix with just mean of diagonal
    target = diag_mean * np.eye(S.shape[0])
        
    return target


# Cross-validation for sample covariance matrix
def calc_cv_cov(X, betas, n_splits = 15, p = 2, clip = False, sample_weight = None):
    
    # Get shape of X
    T, N = X.shape
    
    if sample_weight is None:
        
        sample_weight = np.ones(T)
        
    else:
        
        sample_weight = np.asarray(sample_weight)

    # Create array to hold errors
    errors = np.zeros(shape = betas.shape)
    
    cv = KFold(n_splits = n_splits).split(X)
    
    # Loop over the number of folds
    for train_idx, test_idx in cv:
        
        # Compute clipped or sample covariance matrix for the training values
        S_sample = get_clip_cov(X[train_idx, :]) if clip else np.cov(X[train_idx, :], rowvar = False)
        
        # Get the shrinkage target for the sample
        S_target = get_cov_target(S_sample)
            
        # Calculate the sample covariance matrix using only observations within the fold 
        S_test = get_clip_cov(X[test_idx, :]) if clip else np.cov(X[test_idx, :], rowvar = False)
        
        # Calculate weight given to fold
        wt = np.sum(sample_weight[test_idx])
        
        errors += np.array(
            [wt * np.sum(np.abs(S_test - beta * S_sample - (1 - beta) * S_target)**p) for beta in betas])
    
    # Compute clipped or sample covariance matrix for the entire data frame
    S_sample = get_clip_cov(X) if clip else np.cov(X, rowvar = False)  
  
    # Get the shrinkage target for the whole sample
    S_target = get_cov_target(S_sample)
    
    # Get the optimal beta           
    beta = betas[np.argmin(errors)]
    
    # Calculate the estimated covariance 
    Sigma = beta * S_sample + (1 - beta) * S_target
    
    return Sigma


# Create function that returns the shrunken mean after cross-validation
def calc_cv_mean(X, betas, n_splits = 15, p = 2, sample_weight = None):
    
    if sample_weight is None:
        
        sample_weight = np.ones(X.shape[0])
        
    else:
            
        sample_weight = np.asarray(sample_weight)
        
    cv = KFold(n_splits = n_splits).split(X)
    
    # Create a data frame to hold the results
    errors = np.zeros(shape = betas.shape)
    
    # Loop over the number of folds
    for train_idx, test_idx in cv:
        
        X_train = X[train_idx, :]
        
        # Calculate the sample mean using only observations within the fold   
        m_test = X[test_idx, :].mean()
        
        # Calculate weight given to fold
        wt = np.sum(sample_weight[test_idx])
        
        # Calculate errors
        errors += np.array(
            [wt * np.sum(np.abs(m_test - ledoit_mean(X_train, beta))**p) for beta in betas])
        
    # Get the arguments with the lowest average error
    beta = betas[np.argmin(errors)]
        
    # Return the mean estimate
    return ledoit_mean(X, beta)


# Create partial sample regression
class PSR:
    '''
    Partial sample regression. Taken from Kinlaw et. al. "Asset Allocation" Chapter 13
    '''        
        
    def informativeness(self, x, x_bar = None, Omega_inv = None):
        
        if x_bar is None:
            x_bar = self.x_bar
            
        if Omega_inv is None:  
            Omega_inv = self.Omega_inv          
            
        return 0.5 * float((x - x_bar) @ Omega_inv @ (x - x_bar))
    
    
    def similarity(self, x1, x2, Omega_inv = None):

        if Omega_inv is None: 
            
            Omega_inv = self.Omega_inv          
         
        return -0.5 * float((x1 - x2) @ Omega_inv @ (x1 - x2)) 
    
    
    def relevance(self, x1, x2, x_bar = None, Omega_inv = None):
        
        return self.similarity(x1, x2, Omega_inv) + self.informativeness(x1, x_bar, Omega_inv)
    
    
    # Create fit method
    def fit(self, X, y, betas, thresholds, sample_weight = None, N = 20, 
            n_splits = 5, clip = True, class_weight = None):
          
        
        if sample_weight is None:
            
            sample_weight = np.ones(X.shape[0])
            
        else:
            
            sample_weight = np.asarray(sample_weight)
            
        if class_weight is None:
            
            self.class_weight = np.array([1 - np.mean(y), np.mean(y)])
                  
        else:
            
            self.class_weight = np.asarray(class_weight)
        
        # Make y mean 0
        self.y = y
        
        # Record X
        self.X = X
                
        # Calculate the covariance matrix       
        self.Omega = calc_cv_cov(X, betas, n_splits = n_splits, clip = clip)
        
        # Take the inverse
        self.Omega_inv = np.linalg.pinv(self.Omega)
        
        # Calculate x_bar
        self.x_bar = calc_cv_mean(X, betas, n_splits = n_splits)
        
        # Create an array that holds errors
        errors = np.zeros(len(thresholds))
        
        # Calculate folds
        cv = KFold(n_splits = n_splits).split(X)
        
        # Loop over cv
        for train_idx, test_idx in cv:
            
            # Create the train and test sets
            X_train, y_train = X[train_idx, :], y[train_idx]
            X_test, y_test = X[test_idx, :], y[test_idx]
            
            Omega_temp = calc_cv_cov(X, betas, n_splits = n_splits, clip = clip, 
                                     sample_weight = sample_weight)
            
            Omega_temp_inv = np.linalg.pinv(Omega_temp)
            
            x_bar_temp = calc_cv_mean(X, betas, n_splits = 15, p = 2, 
                                     sample_weight = sample_weight)
            
            if class_weight is None:
                
                wt_temp = np.array([1 - np.mean(y_train), np.mean(y_train)])
                
            else:
                
                wt_temp = class_weight
            
            wt = np.sum(sample_weight[test_idx])
            
            y_temp = y_train - np.mean(y_train)
            
            def pred_proba_val(x, threshold):
                
                rels = np.apply_along_axis(lambda val: self.relevance(val, x, x_bar_temp, Omega_temp_inv), 1, X_train)
                
                return wt_temp[1] + np.mean(rels[rels > threshold] * y_temp[rels > threshold])
                
            for i, threshold in enumerate(thresholds):
                
                y_pred = np.apply_along_axis(pred_proba_val, 1, X_test, threshold = threshold)
                
                errors[i] += wt * log_loss(y_test, y_pred)
                
        self.threshold = thresholds[np.argmin(errors)]


    # Create method to produce the probabilities
    def predict_proba(self, X):
    
        # Initialize list to save results
        probs = []
        
        y_temp = self.y - np.mean(self.y)
        
        def pred_proba_val(x):
            
            rels = np.apply_along_axis(lambda val: self.relevance(val, x), 1, self.X)
            
            return self.class_weight[1] + np.mean(rels[rels > self.threshold] * y_temp[rels > self.threshold])
        
        # Initialize array
        probs = np.zeros(shape = (X.shape[0], 2))
        
        # Calculate the probablity of label 1
        probs[:, 1] = np.apply_along_axis(pred_proba_val, 1, X)
        
        # Calculate the probability of label 0
        probs[:, 0] = 1 - probs[:, 1]
        
        return probs
    
    
    # Use above method to create method to predict log of probabilities
    def predict_log_proba(self, X):
        
        return np.log(self.predict_proba(X))

    
    # Create method to predict the label
    def predict(self, X):
                   
       # Predict probabilities
       probas = self.predict_proba(X)
    
       # Subset to the second column
       probas = probas[:, 1].reshape(newshape = (-1, ))
       
       # Initiliaze predictions
       preds = np.zeros(shape = probas.shape)
       
       # Everything greater than 0.5 is 1
       preds[probas > 0.5] = 1
       
       # Everything less than or equal to 0.5 is 0
       preds[probas <= 0.5] = 0 
       
       return preds
 
    
    def score(self, X, y):
        
        return np.mean(y == self.predict(X))