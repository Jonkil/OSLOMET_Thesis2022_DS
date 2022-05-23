import numpy as np
import pandas as pd
import sys
from mne.io import read_raw_edf
import mne

def read_edf_to_pandas(edf_filename, select_channels = True):
    """ Reads data from an edf file to a Pandas dataframe.
        Column names are 'channel_labels'.
        
        If 'select_channels=True', then only 19 common channels are selected to 
        create the resulting dataframe. The channel names will be updated (standardized).
        
        Returns: dataframe, channel labels
    """
    # read edf file
    raw_data = read_raw_edf(edf_filename, verbose=False, preload=False)
    
    if select_channels:
        # the TUEP database has 3 EEG channel configurations: '02_tcp_le', '03_tcp_ar_a', '01_tcp_ar'
        # number of channels and channel names differ within these configurations
        # to be able to compare the different EEG readings we need to select channels
        # that are common for all configurations

        # the list of 19 channels (their short labels) that we will use for analysing EEG data
        channels_to_use = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8',
                           'T3', 'C3', 'CZ', 'C4', 'T4', 'T5',
                           'P3', 'PZ', 'P4', 'T6', 'O1', 'O2']
        
        # the function to update channel names from original to new format:
        ch_name_update_func = lambda ch: ch.split(' ')[-1].split('-')[0]

        # renaming the original channel names in one .edf file;
        # the update will be written into the in-memory edf object
        raw_data.rename_channels(mapping=ch_name_update_func)
        
        # check if all required channels are in the edf file
        try:
            assert all([ch in raw_data.info["ch_names"] for ch in channels_to_use])
        except:
            print('Not all required channels are in the edf file.')
        
        # dataframe with EEG readings from selected channels and with 
        # updated channel names
        df = pd.DataFrame(raw_data.pick_channels(channels_to_use).get_data().T,
            columns=raw_data.pick_channels(channels_to_use).info['ch_names'])
        
        # we need to return correct channel/column names
        channel_labels = channels_to_use # as specified by us: left-to-right and top-down
        # channel_labels = df.columns.tolist() # as given in the edf file
        
    else:
        # get channel names from edf file
        channel_labels = raw_data.info["ch_names"]

        # create a dataframe from
        df = pd.DataFrame(raw_data.get_data().T, columns=channel_labels)

    return df[channel_labels], channel_labels # as specified by us: left-to-right and top-down


def entropy(bins, *X):
    
    # binning of the data
    data, *edges = np.histogramdd(X, bins=bins)
    
    # calculate probabilities
    data = data.astype(float)/data.sum()
    
    # compute H(X,Y,...,Z) = sum(-P(x,y,...,z) ∗ log2(P(x,y,...,z)))
    return np.sum(-data * np.log2(data+sys.float_info.epsilon))


def mutual_information(bins, X, Y):
    
    # compute I(X,Y) = H(X) + H(Y) − H(X,Y)
    
    H_X = entropy(bins, X)
    H_Y = entropy(bins, Y)
    H_XY = entropy(bins, X, Y)
    
    return H_X + H_Y - H_XY

# Compute number of bins using Sturge's rule
def compute_mi_matrix(df):
    """ Compute Mutual Information matrix.
    
        Return: mi_matrix
    """
    n_cols = df.shape[1]
    mi_matrix = np.zeros([n_cols, n_cols])
    
    # Sturge's rule for number of bins
    n_bins = int(1 + 3.322*np.log2(df.shape[0]))
    
    for i in range(n_cols):
        for j in range(n_cols):
            mi = mutual_information(n_bins, df.iloc[:,i],df.iloc[:,j])
            mi_matrix[i,j] = mi
    
    return mi_matrix
    

def filter_data(filename, l_freq, h_freq):

    raw = read_raw_edf(filename, verbose=0, preload=False)
    sfreq = raw.info['sfreq']
    
    df, ch = read_edf_to_pandas(filename)
    df_filt = mne.filter.filter_data(data=df.T.values, sfreq=sfreq,
                                    l_freq=l_freq, h_freq=h_freq,
                                    verbose=False)
    
    return pd.DataFrame(df_filt.T, columns=ch)


def compute_normed_mi_matrix(mi_matrix):
    """ Compute normalized version of the given Mutual Information matrix.
    
        Return: normed_mi_matrix
    """
    
    # normalize mi matrix by dividing matrix elements with
    # sqrt of product of respective diagonal elements
    divisor_matrix = np.sqrt(np.diag(mi_matrix)*np.diag(mi_matrix).reshape(-1,1))
    normed_mi_matrix = mi_matrix/divisor_matrix

    return normed_mi_matrix

def compute_mi_over_bands_form_edf_file(edf_file):
    freq_bands = {'all':(1,40), 
                  'delta':(1,4),
                  'theta': (4,8),
                  'alpha': (8,12),
                  'beta': (12,30)
                  }
    results = {}

    for k in freq_bands.keys():
        (l_freq,h_freq) = freq_bands[k]
        df = filter_data(edf_file, l_freq=l_freq, h_freq=h_freq)
        corr_matrix = df.corr()
        mi_matrix = compute_mi_matrix(df=df)
        normed_mi_matrix = compute_normed_mi_matrix(mi_matrix)
        results[k] = (normed_mi_matrix, corr_matrix)    

    return edf_file, results