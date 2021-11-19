"""
Date 11/11/2021
Autor: Alex Treacher

This script is designed use the final trained MEGnet to make predictions on data.
It is set up to be ran from the command line.

Note: Tensroflow does take some time to load, thus running this independently for each subject is not the most computationally efficient.
    To increase efficeny, I'd suggest imbedding this function into a pipeline that will load tensorflow and then run multiple subjects at once
    Alternativley, fPredictChunkAndVoting (used in function below) can be applied to N spatial map and time series pairs. 
    Thus the fPredictICA could be easily modified to be appled to a complete list of ICA components and ran on many subjects.

The outputs are saved by numpy in a text file, that is easliy human readable and can be loaded using np.loadtxt('/path/to/ICA_component_lables.txt')

example usage:
python label_ICA_components.py --input_path example_data/HCP/100307/@rawc_rfDC_8-StoryM_resample_notch_band/ICA202DDisc --output_dir example_data/HCP/100307/@rawc_rfDC_8-StoryM_resample_notch_band/ICA202DDisc --output_type list
"""


import os
import sys
import argparse
import numpy as np
import pandas as pd
import scipy.io


import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa

import megnet_utilities
from megnet_utilities import fPredictChunkAndVoting

def fPredictICA(strSubjectICAPath, strOutputDir=None, strOutputType='list', strModelPath = 'model/MEGnet_final_model.h5'):
    #loading the data is from our Brainstorm Pipeline, it may require some minor edits based on how the data is saved.
    #load the time seris and the spatial map
    arrTimeSeries = scipy.io.loadmat(os.path.join(strSubjectICAPath,'ICATimeSeries.mat'))['arrICATimeSeries'].T
    arrSpatialMap = np.array([scipy.io.loadmat(os.path.join(strSubjectICAPath,f'component{i}.mat'))['array'] for i in range(1,21)])
    #crop the spatial map to remove additional pixels
    arrSpatialMap = arrSpatialMap[:,30:-30,15:-15,:]

    #ensure the data is compatable
    try:
        assert arrTimeSeries.shape[0] == arrSpatialMap.shape[0] #the number of time series should be the same as the number of spatial maps
        assert arrSpatialMap.shape[1:]==(120,120,3) #the spatial maps should have a shape of [N,120,120,3]
        assert arrTimeSeries.shape[1] >= 15000 #the time series need to be at least 60secs with a sample rate of 250hz (60*250=15000)
    except AssertionError:
        raise ValueError('The data does not have the correct dimsions')
        

    #load the model
    kModel = keras.models.load_model(strModelPath)

    #use the vote chunk prediction function to make a prediction on each input
    output = fPredictChunkAndVoting(kModel, 
                                    arrTimeSeries, 
                                    arrSpatialMap, 
                                    np.zeros((20,3)), #the code expects the Y values as it was used for performance, just put in zeros as a place holder.
                                    intModelLen=15000, 
                                    intOverlap=3750)
    arrPredicionsVote, arrGTVote, arrPredictionsChunk, arrGTChunk = output

    #format the predictions
    if strOutputType.lower() == 'array':
        to_return = arrPredicionsVote[:,0,:]
    else:
        to_return = arrPredicionsVote[:,0,:].argmax(axis=1)
    
    #save the predictions if path is given
    if not strOutputDir is None:
        strOutputPath = os.path.join(strOutputDir,'ICA_component_lables.txt')
        np.savetxt(strOutputPath, to_return)

    return to_return

if __name__ == "__main__":
    #load the arguments and print the inputted path
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_path', dest='strSubjectICAPath', type=str, help='Path to the folder that contains the ICA components for a subject')
    parser.add_argument('-o', '--output_dir', dest='strOutputDir', type=str, help='Path to the output directory to save the labels')
    parser.add_argument('-t', '--output_type', dest='strOutputType', type=str, default='list', help='list (default) or array. If list, the output will be a list with predictions of the components. If array, the one hot encoded probabilites of the prediction will be outputted')

    args = parser.parse_args()

    print(f'Predicting on {args.strSubjectICAPath}')
    fPredictICA(args.strSubjectICAPath, args.strOutputDir, args.strOutputType)