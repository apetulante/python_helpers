#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:27:03 2017

@author: petulaa
"""

from sklearn.preprocessing import RobustScaler
import numpy as np
from sklearn.cross_validation import train_test_split

m_part = 3.3e7


def construct_Xy(Xfull, yfull, Xrange, *args):
    """
     Take a matrix of all features, and take just a portion, or bin it by
     another parameter to create filtered X/y
     
     *args are :
         - binning_param (the thing to be binned by)
         - bin_range (upper and lower bin bounds)
    """
    if len(args) != 0:
        binning_param = args[0]
        bin_range = args[1]
        bin_delete = np.where(binning_param<bin_range[0])[0]
        bin_delete = np.append(bin_delete,np.where(binning_param>bin_range[1])[0])
        
    X = []    
    for i in range(Xrange[0], Xrange[1]):  
        if len(args) != 0:
            X.append(np.delete(Xfull[i], bin_delete))
        else:
            X.append(Xfull[i])
    if len(args) != 0:
        y = np.delete(yfull, bin_delete)
    else:
        y = yfull
        
    X = np.transpose(X)
    
    return X, y


def get_accuracy(model, X, y, test_size, accuracy_type, tolerance,
                 accuracy_param=None):    
    """
    Take any scikit-learn model that has a fit() and predict() method,
    and get an accuracy of predictions, given some tolerance in the final
    values. 
    Accuracy_type is the way in which to define accuracy of the model.
    Valid accuracy types are:
        - frac_diff: Prediction is correct if fractional difference abs(true-pred)/true
          is less than the tolerance
        - diff: Prediction is correct if the absolute difference abs(true-pred) is less than
          the tolerance
        - norm_diff: Prediction is correct if the absolute difference abs(true-pred)/param
          normalized by another parameter (param) is less than the tolerance. The normed
          parameter must be already a row of the X feature-matrix
        - oom: Prediction is correct if it's within some tolerance*the order of magnitude
          of the truth
        - weight_diff: Prediction is correct if the difference between true and prediction is less than
          some tolerance times the difference between true and a weight parameter
    """    
    X = np.hstack((X,accuracy_param.reshape(len(accuracy_param),1)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)     
    norm = X_test[:,-1]
    X_test = X_test[:,0:-1]
    X_train = X_train[:,0:-1]
    X_test_unscaled = X_test
    
    scaler_x = RobustScaler()
    scaler_x.fit(X_train)    
    X_train = scaler_x.transform(X_train)
    X_test = scaler_x.transform(X_test)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    
    if accuracy_type == 'frac_diff':
        num_correct=0
        for i in range(len(predictions)):
            if np.abs(predictions[i]-y_test[i])/y_test[i] < tolerance:
                num_correct += 1
        prediction_accuracy = num_correct/len(y_test)
    
    elif accuracy_type == 'diff':
        num_correct=0
        for i in range(len(predictions)):
            if np.abs(predictions[i]-y_test[i]) < tolerance:
                num_correct += 1
        prediction_accuracy = num_correct/len(y_test)
        
    elif accuracy_type == 'norm_diff':
        num_correct=0
        for i in range(len(predictions)):
            if np.abs(predictions[i]-y_test[i])/norm[i] < tolerance:
                num_correct += 1
        prediction_accuracy = num_correct/len(y_test)
    
    elif accuracy_type == 'oom':
        num_correct=0
        for i in range(len(predictions)):
            if np.abs(predictions[i]-y_test[i]) < (tolerance*(10**int(np.log10(norm[i])))):
                num_correct += 1
        prediction_accuracy = num_correct/len(y_test)
    
    elif accuracy_type == 'weight_diff':
        num_correct=0
        for i in range(len(predictions)):
            if np.abs(predictions[i]-y_test[i]) < (tolerance*(np.abs(norm[i]-y_test[i]))):
                num_correct += 1
        prediction_accuracy = num_correct/len(y_test)
    
    return prediction_accuracy



def get_predictions(model, X, y, test_size, accuracy_type, accuracy_param=None,
                    return_test = False):
    """
    Same as get_accuracy, except actually return all of the predictions and the
    comparison values (to be compared to a tolerance) directly
    Accuracy_type is the way in which to define accuracy of the model.
    Valid accuracy types are:
        - frac_diff: Prediction is correct if fractional difference abs(true-pred)/true
          is less than the tolerance
        - diff: Prediction is correct if the absolute difference abs(true-pred) is less than
          the tolerance
        - norm_diff: Prediction is correct if the absolute difference abs(true-pred)/param
          normalized by another parameter (param) is less than the tolerance. The normed
          parameter must be already a row of the X feature-matrix
        - oom: Prediction is correct if it's within some tolerance*the order of magnitude
          of the truth
        - weight_diff: Prediction is correct if the difference between true and prediction is less than
          some tolerance times the difference between true and a weight parameter
"""
    X = np.hstack((X,accuracy_param.reshape(len(accuracy_param),1)))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size)     
    norm = X_test[:,-1]
    X_test = X_test[:,0:-1]
    X_train = X_train[:,0:-1]
    X_test_unscaled = X_test
    
    scaler_x = RobustScaler()
    scaler_x.fit(X_train)    
    X_train = scaler_x.transform(X_train)
    X_test = scaler_x.transform(X_test)
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    
    if accuracy_type == 'frac_diff':
        prediction_accuracies = np.abs(predictions-y_test)/y_test
    
    elif accuracy_type == 'diff':
        prediction_accuracies = np.abs(predictions-y_test)
        
    elif accuracy_type == 'norm_diff':
        prediction_accuracies = np.abs(predictions-y_test)/norm
    
    elif accuracy_type == 'oom':
        prediction_accuracies = np.abs(predictions-y_test)/(10**np.log10(y_test).astype(int))
    
    elif accuracy_type == 'weight_diff':
        prediction_accuracies = np.abs(predictions-y_test)/(np.abs(norm-y_test))
    
    if return_test == True:
        return predictions, prediction_accuracies, X_test_unscaled, y_test, norm
    else:
        return predictions, prediction_accuracies
        
    
    
