

import keras.backend as K
import numpy as np
from scipy import stats
import logging
import tensorflow as tf


def nse_loss(y_true, y_pred):
    y_true = y_true[:, 365:, :]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, 365:, :]  # Omit values in the spinup period (the first 365 days)

    numerator = K.sum(K.square(y_pred - y_true), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)

    return numerator / denominator


def nse_metrics(y_true, y_pred):
    y_true = y_true[:, 365:, :]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, 365:, :]  # Omit values in the spinup period (the first 365 days)

    numerator = K.sum(K.square(y_pred - y_true), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)
    rNSE = numerator / denominator

    return 1.0 - rNSE

def rmse_loss(y_true, y_pred):
    y_true = y_true[:, 365:, :]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, 365:, :]  # Omit values in the spinup period (the first 365 days)
    
    mse=np.sum((y_pred-y_true)**2)/len(y_true)
    rmse=mse**0.5
    
    return 1.0-rmse

def rmse(y_true, y_pred):
    y_true = y_true[:, 365:, :]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, 365:, :]  # Omit values in the spinup period (the first 365 days)
    
    mse=np.sum((y_pred-y_true)**2)/len(y_true)
    rmse=mse**0.5
    
    return rmse

def Krmse_loss(y_true, y_pred):
    y_true = y_true[:, 365:, :]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, 365:, :]  # Omit values in the spinup period (the first 365 days)
    
    mse=K.sum(K.square(y_pred-y_true), axis=1)/len(y_true)
    rmse=K.root(mse)
    
    return 1.0-rmse
    
def Krmse(y_true, y_pred):
    y_true = y_true[:, 365:, :]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, 365:, :]  # Omit values in the spinup period (the first 365 days)
    
    mse=K.sum(K.square(y_pred-y_true), axis=1)/len(y_true)
    rmse=K.root(mse)
    
    return rmse





    

