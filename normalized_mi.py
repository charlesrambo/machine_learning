# -*- coding: utf-8 -*-
"""
Created on Sun Feb  8 19:22:40 2026

@author: cramb
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif

def get_num_bins(num_obs, corr=None):
    """
    Optimal number of bins for discretization based on Hacine-Gharbi et al.
    
    Univariate case: Analytical solution to the cubic equation for marginal entropy.
    Bivariate case: Scales bins based on correlation to capture joint information.
    """
    # Univariate case
    if corr is None:
        zeta = np.cbrt(8 + 324 * num_obs + 12 * np.sqrt(36 * num_obs + 729 * num_obs**2))
        bins = np.round(zeta/6 + 2/(3 * zeta) + 1/3)
        
    # Bivariate case
    else:
        # Clip correlation so the number of bins doesn't explode
        corr = np.clip(corr, -0.95, 0.95)
        bins = np.round(np.sqrt(1/2) * np.sqrt(1 + np.sqrt(1 + 24 * num_obs/(1 - corr**2))))
        
    return int(np.max([bins, 2]))


def preprocess_for_mi(df):
    """
    Fills, factorizes, and discretizes the entire dataframe once
    using optimized marginal binning (univariate case).
    """
    df_discrete = df.copy()
    
    # Calculate the universal optimal bin count for this sample size
    bins = get_num_bins(len(df_discrete))
    
    for col in df_discrete.columns:
        # Handle Categorical / Boolean / Object
        if df_discrete[col].dtype == 'object' or \
           df_discrete[col].dtype.name == 'category' or \
           pd.api.types.is_bool_dtype(df_discrete[col]):
            
            df_discrete[col], _ = df_discrete[col].factorize()
            df_discrete[col] = df_discrete[col].fillna(-1)
            
        # Handle Continuous (Float/Int)
        else:
            # Impute with mean
            df_discrete[col] = df_discrete[col].fillna(df_discrete[col].mean())
            
            # Use Rank-based Quantile binning to maximize entropy
            # rank(method='first') ensures no ties, preventing qcut errors
            df_discrete[col] = pd.qcut(df_discrete[col].rank(method='first'), 
                                       q=bins, labels=False)
            
    return df_discrete


def get_nmi(df, x_cols, y_col, random_state=0, n_jobs=-1):
    """
    Calculates Normalized Mutual Information (NMI) on a discretized data frame.
    Normalized by the Shannon Entropy of the target variable.
    """
    X = df[x_cols]
    y = df[y_col]
    
    # Calculate Shannon Entropy of the target for normalization
    # This keeps results bounded between 0 and 1
    target_entropy = entropy(y.value_counts(normalize=True))
    
    if target_entropy == 0:
        return pd.Series(0.0, index=x_cols)
    
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y, discrete_features=True, 
                                    random_state=random_state, 
                                    n_jobs=n_jobs)
    
    # Create NMI series
    nmi_series = pd.Series(mi_scores/target_entropy, index=x_cols)
    
    return nmi_series


def create_nmi_df(df, n_jobs = -1):
    """
    Systematically calculates the NMI matrix for all columns.
    """
    
    # Discretize the whole data frame once
    df_preped = preprocess_for_mi(df)
    
    # Initialize dictionary to hold results
    nmi_results = {}
    
    # Iterate through targets
    for y_col in df_preped:
        
        print(f"Analyzing Target: {y_col}")
        
        x_cols = [col for col in df_preped if col != y_col]
        
        nmi_results[y_col] = get_nmi(df_preped, x_cols, y_col, n_jobs = n_jobs)
    
    # Create DataFrame and fill diagonal with 1.0
    final_df = pd.DataFrame(nmi_results).fillna(1.0)
    
    # Sort the final result
    final_df = final_df.loc[final_df.index, final_df.index]
    
    return final_df

if __name__ == "__main__":
    # Quick Example Usage:
    # data = pd.read_csv('your_data.csv')
    # nmi_matrix = create_nmi_df(data)
    # print(nmi_matrix)
    pass