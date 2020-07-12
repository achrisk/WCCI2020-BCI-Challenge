from scipy.io import loadmat
from scipy.signal import butter, lfilter

import numpy as np

import math

from pyriemann.estimation import Covariances, Coherences
from pyriemann.tangentspace import TangentSpace, FGDA

sample_rate = 512 #in Hz
num_channels = 12
path = './db/'

# load train data
def load_train_data(path_to_file):
    annots = loadmat(path_to_file)
    raw_eeg_data = annots['RawEEGData']
    labels = annots['Labels']

    return raw_eeg_data, labels

# load evaluate data
def load_eval_data(path_to_file):
    annots = loadmat(path_to_file)
    raw_eeg_data = annots['RawEEGData']

    return raw_eeg_data

def get_coh(X):
    coh = Coherences().transform(X)
    return coh[:,:,:,0]

def get_covar(X):
    covar = Covariances(estimator='lwf').transform(X)
    return covar

def get_riemann_ts(X):
    covar = Covariances(estimator='lwf').transform(X)
    tang = TangentSpace().fit_transform(covar)
    return tang

# Butterworth Bandpass Filter
# Source: https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(signal, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    n_trials, n_chans, n_time = signal.shape
    filtered = np.zeros(signal.shape)
    for i in range(n_trials):
        for j in range(n_chans):
            filtered[i][j] = lfilter(b, a, signal[i][j])
    return filtered

def get_data_within(patient_number, trim=(-1536,0), low=8, high=24, order=5):
    if patient_number<1 or patient_number>10:
        print('Invalid Patient Number')
        return None
    
    print('Mode = within')
    if patient_number<10:
        patient_str = '0'+str(patient_number)
    else:
        patient_str = str(patient_number)

    tlow, thigh = trim
    raw_data, labels = load_train_data(path+'parsed_P'+patient_str+'T.mat')
    if thigh==0:
        raw_data = raw_data[:,:,tlow:]
    else:
        raw_data = raw_data[:,:,tlow:thigh]
    labels = np.squeeze(labels)
    labels -= 1
    y = labels

    raw_eval = load_eval_data(path+'parsed_P'+patient_str+'E.mat')
    if thigh==0:
        raw_eval = raw_eval[:,:,tlow:]
    else:
        raw_eval = raw_eval[:,:,tlow:thigh]

    print('Loaded data for patient %d'%(patient_number))

    #filter data
    X = butter_bandpass_filter(raw_data, low, high, sample_rate, order=order)

    #filter evaluation data
    X_eval = butter_bandpass_filter(raw_eval, low, high, sample_rate, order=order)

    return X, y, X_eval

def get_data_inter(patient_number, trim=(-1536,0), low=8, high=24, order=5, mode="test"):
    if patient_number<1 or patient_number>10:
        print('Invalid Patient Number')
        return None

    tlow, thigh = trim
    print('Mode = inter')
    raw_train = np.array([])
    y_train = np.array([])
    for i in range(1,11):
        if i!=patient_number and i!=9 and i!=10:
            if i<10:
                patient_str = '0'+str(i)
            else:
                patient_str = str(i)
            raw_data_train, labels = load_train_data(path+'parsed_P'+patient_str+'T.mat')
            if thigh==0:
                raw_data_train = raw_data_train[:,:,tlow:]
            else:
                raw_data_train = raw_data_train[:,:,tlow:thigh]
            labels = np.squeeze(labels)
            labels -= 1
            print('Loaded data for patient %d'%(i))

            if raw_train.size==0:
                raw_train = raw_data_train
                y_train = labels
            else:
                raw_train = np.concatenate((raw_train, raw_data_train))
                y_train = np.concatenate((y_train, labels))

    if patient_number<10:
        patient_str = '0'+str(patient_number)
    else:
        patient_str = str(patient_number)

    if mode=="test":
        raw_test, labels = load_train_data(path+'parsed_P'+patient_str+'T.mat')
        if thigh==0:
            raw_test = raw_test[:,:,tlow:]
        else:
            raw_test = raw_test[:,:,tlow:thigh]
        labels = np.squeeze(labels)
        labels -= 1
        y_test = labels

    raw_eval = load_eval_data(path+'parsed_P'+patient_str+'E.mat')
    if thigh==0:
        raw_eval = raw_eval[:,:,tlow:]
    else:
        raw_eval = raw_eval[:,:,tlow:thigh]

    print('Loaded data for patient %d'%(patient_number))

    #filter training data
    X_train = butter_bandpass_filter(raw_train, low, high, sample_rate, order=order)

    if mode=="test":
        #filter testing data
        X_test = butter_bandpass_filter(raw_test, low, high, sample_rate, order=order)

    #filter evaluation data
    X_eval = butter_bandpass_filter(raw_eval, low, high, sample_rate, order=order)

    if mode=="test":
        return X_train, y_train, X_test, y_test, X_eval
    
    return X_train, y_train, X_eval

def mat_to_vect(mat):
    mat_size = mat.shape[0]
    vect_size = mat_size*mat_size
    vect = np.zeros(vect_size)
    idx=0
    for i in range(mat_size):
        for j in range(mat_size):
            vect[idx] = mat[i][j]
            idx+=1
    return vect

def symm_to_vect(mat):
    mat_size = mat.shape[0]
    vect_size = int((mat_size*(mat_size+1))/2)
    vect = np.zeros(vect_size)
    idx=0
    for i in range(mat_size):
        for j in range(i, mat_size):
            vect[idx] = mat[i][j]
            idx+=1
    return vect

def extract_covar_windows(series, win_size, hop_size):
    channels, num_samples = series.shape
    num_wins = int(math.ceil((num_samples-win_size)/hop_size))+1
    time_wins = np.zeros((num_wins, channels, win_size))
    covar_size = int((channels*(channels+1))/2)
    covar_wins = np.zeros((covar_size, num_wins))
    for i in range(num_wins):
        if i<num_wins-1:
            time_wins[i,:,:] = series[:,i*hop_size:i*hop_size+win_size]
        else:    
            time_wins[i,:,:num_samples-i*hop_size] = series[:,i*hop_size:num_samples]
    covar = get_covar(time_wins)
    for i in range(num_wins):
        covar_wins[:,i] = symm_to_vect(covar[i])
    return covar_wins

def extract_riemann_windows(series, win_size, hop_size):
    channels, num_samples = series.shape
    num_wins = int(math.ceil((num_samples-win_size)/hop_size))+1
    time_wins = np.zeros((num_wins, channels, win_size))
    covar_size = int((channels*(channels+1))/2)
    for i in range(num_wins):
        if i<num_wins-1:
            time_wins[i,:,:] = series[:,i*hop_size:i*hop_size+win_size]
        else:    
            time_wins[i,:,:num_samples-i*hop_size] = series[:,i*hop_size:num_samples]
    covar = np.transpose(get_riemann_ts(time_wins))
    return covar