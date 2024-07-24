# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 07:17:35 2024

@author: charlesr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Use Seaborn style
plt.style.use("seaborn-v0_8")

# Define function
def get_strat_distributions(df, model, wt_fun, signals, target, rtn, date, cv,
                            freq = 1, sample_weight = None, title = None, 
                            filename = None, bins = None, **kwargs):
    
    # Get the number of splits
    n_splits = cv.get_n_splits(X = df)
    
    # Initialize dictionary to hold simulated results
    sim_dict = {'Sharpe':np.zeros(n_splits), 'STD':np.zeros(n_splits), 
                    'IR':np.zeros(n_splits), 'Spearman':np.zeros(n_splits)}
    
    # Iterate over splits
    for fold, (train_idx, test_idx) in enumerate(cv.split(df)):
        
        # Make a deepcopy of df
        df_copy = df.copy()
        
        # Fit model on training data
        model = model.fit(df_copy.loc[train_idx, signals].values, 
                          df_copy.loc[train_idx, target].values, 
                          sample_weight = sample_weight.loc[train_idx] if sample_weight is not None else None)
        
        # Subset to test data
        df_copy = df_copy.loc[test_idx, :]
    
        # Calculate market return on test data
        mkt_arr = df_copy.groupby(date)[rtn].mean().values    
        
        # Get probability of 1
        df_copy['prob'] = model.predict_proba(df_copy.loc[test_idx, signals].values)[:, 1]
        
        # Calculate weight
        df_copy['wt'] = df_copy.groupby('Date')['prob'].transform(lambda x: wt_fun(x))
        
        # Multiply wt by return
        df_copy[rtn] *= df_copy['wt']
        
        # Compute the return for each period in our test set
        rtn_arr = df_copy.groupby(date)[rtn].sum().values
        
        # Calculate stats    
        sim_dict ['IR'][fold] = np.sqrt(freq) * np.mean(rtn_arr - mkt_arr)/np.std(rtn_arr - mkt_arr)
        sim_dict['Sharpe'][fold] = np.sqrt(freq) * np.mean(rtn_arr)/np.std(rtn_arr)
        sim_dict['STD'][fold] = np.sqrt(freq) * np.std(rtn_arr)
        sim_dict['Spearman'][fold] = spearmanr(rtn_arr, mkt_arr)[0]
        
    # Initialize data frame to hold results   
    results = pd.DataFrame(index = list(sim_dict.keys()), 
                               columns = ['median', 'Q1', 'Q3', 'mean', 
                                          'std_error', 't-stat'])
        
    # Loop over key
    for key in sim_dict:
            
        # Calculate median
        results.loc[key, 'median'] = np.nanmedian(sim_dict[key]) 

        # Calculate Q1
        results.loc[key, 'Q1'] = np.nanquantile(sim_dict[key], 0.25) 

        # Calculate Q3
        results.loc[key, 'Q3'] = np.nanquantile(sim_dict[key], 0.75) 
            
        # Calculate mean
        results.loc[key, 'mean'] = np.nanmean(sim_dict[key])
            
        # Calculate standard error
        results.loc[key, 'std_error'] = np.nanstd(sim_dict[key])/np.sqrt(len(sim_dict[key]))
                
    # Calculate the t-statistic
    results['t-stat'] = results['mean']/results['std_error'] 
        
    # Initialize subplots
    fig, ax = plt.subplots(2, 2, **kwargs)
    
    if bins is None:
        
        bins = int(np.sqrt(n_splits)) if n_splits > 100 else int(n_splits/4)
      
    # Loop over subplots
    for i, key in enumerate(sim_dict):
        
        # Calculate row and column
        row, col = i//2, i % 2
        
        # Plot histogram
        ax[row, col].hist(sim_dict[key], bins = bins, density = True)
        
        # Plot vertical line at Q1
        ax[row, col].axvline(np.quantile(sim_dict[key], 0.25), color = 'blue', 
                             linestyle = 'dashed', linewidth = 1,
                             label = f'Q1: {np.quantile(sim_dict[key], 0.25) :.2f}')
        
        # Plot vertical line at median
        ax[row, col].axvline(np.median(sim_dict[key]), color = 'black', 
                             linestyle = 'dashed', linewidth = 1,
                             label = f'Median: {np.median(sim_dict[key]):.2f}')
 
        # Plot vertical line at Q3
        ax[row, col].axvline(np.quantile(sim_dict[key], 0.75), color = 'blue', 
                             linestyle = 'dashed', linewidth = 1,
                             label = f'Q3: {np.quantile(sim_dict[key], 0.75):.2f}')
        
        # Give plot a title
        ax[row, col].set_title(key)
        
        # Add legend
        ax[row, col].legend()
        
    if title is not None:
        
        fig.suptitle(title)
    
    # Save figure if filename is not None   
    if filename is not None:
        
        plt.savefig(filename)
    
    # Plot figure
    plt.show()
    
    return results


if __name__ == '__main__':
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection._split import KFold

    from scipy.stats import norm, multivariate_normal
    
    
    # Generate some data to test function
    
    # Define number of securities and observations per security
    securities, size = 50, 1_000
    
    # Define signal labels
    signals = ['1', '2', '3']

    # Initialize data frame
    data = pd.DataFrame()

    # Generate results for each security
    for security in range(securities):

        # Calculate signals
        security_stats = pd.DataFrame(multivariate_normal.rvs(mean = [0, 0, 0],
                                                    cov = [[1, 0.5, 0.25], 
                                                           [0.5, 1.0, -0.1],
                                                           [0.25, -0.1, 1.0]],
                                                    size = size), 
                                      columns = signals)
        
        # This is the true relationship between signals and returns
        security_stats['rtn'] = (0.01 * security_stats[signals[0]] 
                                 - 0.005 * security_stats[signals[1]] 
                                 + 0.003 * security_stats[signals[2]] 
                                 + norm.rvs(loc = 0, scale = 0.01, size = size))
        
        # Create date labels
        security_stats['Date'] = np.arange(0, size)
        
        # Add security label
        security_stats['Security'] = security
        
        # Concatenate with results for previous securities
        data = pd.concat([data, security_stats], ignore_index = True)

    del security_stats

    # Create function to convert from probabilities to weights
    def wt_fun(probs):
        
        # Make probabilities 1D array
        probs = np.asarray(probs).flatten()
        
        # Truncate probabilities
        probs[probs > 0.99] = 0.99
        probs[probs < 0.01] = 0.01
        
        # Covert from probabilities to weights
        wt = norm.ppf(probs)
        
        # If there are positive weights...
        if np.sum(wt > 0) > 0:
            
            # ... rescale so they sum to 1
            wt[wt > 0] /= np.sum(wt[wt > 0])
        
        # If there are negative weights...
        if np.sum(wt < 0) > 0:
            
            # ... rescale so they sum to -1
            wt[wt < 0] /= -np.sum(wt[wt < 0])
                 
        return wt

    # Calculate target
    data['target'] = data.groupby('Date')['rtn'].transform(lambda x: x > x.mean()).astype(int)

    # Initialize CV; probability unrealistically large for slower ML models
    cv = KFold(n_splits = 100)

    # Define model
    model = LogisticRegression()
    
    # Get results
    results = get_strat_distributions(data, model, wt_fun, signals, 'target',  
                                      'rtn', 'Date', cv, dpi = 300, 
                                      figsize = (20, 15), freq = 12, 
                                      sample_weight = None, filename = None)
    