import matplotlib.pyplot as plt
import os
import math
import seaborn as sns
import random
import numpy as np
import tensorflow as tf
from keras.utils import np_utils
from imblearn.under_sampling import NearMiss, AllKNN, RandomUnderSampler
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#%% slice the matrix using discontinuous row index 
def discontSliceRow( matrix, index ):
    outputMatrix = np.zeros( [ len( index ), matrix.shape[ 1 ] ] )
    outputIndex = 0
    for processLine in range( 0, len( matrix ) ):
        if processLine in index:
            outputMatrix[ outputIndex, : ] = matrix[ processLine, : ]
            outputIndex += 1
    return outputMatrix

#%% slice the matrix using discontinuous column index
def discontSliceCol( matrix, index ):
    outputMatrix = np.zeros( [ matrix.shape[ 0 ], len( index ) ] )
    outputIndex = 0
    for processCol in range( 0, matrix.shape[1] ):
        if processCol in index:
            outputMatrix[ :, outputIndex ] = matrix[ :, processCol ]
            outputIndex += 1
    return outputMatrix

#%%
def iter_loadtxt(filename, delimiter=',', skiprows=0, dtype= float):
    def iter_func():
        with open(filename, 'r') as infile:
            for _ in range(skiprows):
                next(infile)
            for line in infile:
                line = line.rstrip().split(delimiter)
                for item in line:
                    yield dtype(item)
        iter_loadtxt.rowlength = len(line)

    data = np.fromiter(iter_func(), dtype=dtype)
    data = data.reshape((-1, iter_loadtxt.rowlength))
    return data

#%% plot the distribution of data, compatatbale with high-dimensional np arrays
def plotInputDistribution( inputM, saveFolder = '' ):
    output = np.reshape( inputM, [ countElements( inputM ) ] )
    fig1 = plt.figure(  )
    ax1 = fig1.gca()
    binwidth = ( max( output ) - min( output ) )/1000
    ax1.hist( output, bins=np.arange( min( output ), max( output ) + binwidth, binwidth ) )
    if saveFolder != '':
        fig1.savefig( saveFolder + '/hist.png' )
        plt.close('all')
        
#%% calculate the number of elements of an high-dimensional tensor
def countElements( inputM ):
    inputShape = inputM.shape
    dim = 1
    for i in inputShape:
        dim *= i
    return dim