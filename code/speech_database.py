# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 17:51:45 2017

@author: Kyle
"""

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


class SpeechDatabase:
        
    # 
    dataType = 'waveform'
    testType = 'crossValidation'
    dataName = 'void'
    
    # raw waveform property
    audioLength = 6
    sampleRate = 16000
    
    # spectrogram property
    timeSteps = 150
    fftBins = 64
    
    # labelPosition 
    labelPosition = 0
    
    # overlap this method for each dataset, no need to specific path and do adaption for each database
    def getDataPathCV( self, fold ):
        filePath = ''
        return filePath
    
    def getDataPathHO( self, targetSet ):
        filePath = ''
        return filePath 
        
    def loadData( self, testFold = 0 ):
        mode = self.testType
        if mode == 'holdOut':
            trainFeature, trainLabel , testFeature, testLabel = self.loadDataHO( balance = True )
            return trainFeature, trainLabel , testFeature, testLabel
        elif mode == 'crossValidation':
            trainFeature, trainLabel , testFeature, testLabel = self.loadData5CV( testFold, balance = True )
            return trainFeature, trainLabel , testFeature, testLabel
        elif mode == 'singleTest':
            trainFeature, trainLabel = self.loadDataSingle(  )
            return trainFeature, trainLabel 
        else:
            print( 'Mode not recognized' )
    
    # load data that use 5 fold cross validation strategy, need an input names which fold is test set
    def loadData5CV( self, testFold, balance = True ):
        
        folderList = [ 0, 1, 2, 3, 4 ]
        trainFolderList = folderList.copy( )
        
        # remove the test fold from the training set
        del trainFolderList[ testFold ]
        
        fold = [ 0, 0, 0, 0, 0 ]
        for i in folderList:
            dataPath = self.getDataPathCV( testFold )
            fold[ i ] = iter_loadtxt( dataPath )
        
        # seperate training and testing data
        trainData = eval( 'np.concatenate( ( fold[ ' + str( trainFolderList[ 0 ] ) + \
                                          ' ], fold[ ' + str( trainFolderList[ 1 ] ) + \
                                          ' ], fold[ ' + str( trainFolderList[ 2 ] ) + \
                                          ' ], fold[ ' + str( trainFolderList[ 3 ] ) + ' ] ), axis=0 )' )
        testData = eval( 'fold[ ' + str( testFold ) + ' ]' )
                
        trainFeature, trainLabel = self.processData( trainData, 'train' )
        if balance == True:
            trainFeature, trainLabel = self.balanceData( trainFeature, trainLabel )
        
        testFeature, testLabel = self.processData( testData, 'test' )
        
        return trainFeature, trainLabel , testFeature, testLabel
    
    # load data that use hold-out test strategy
    def loadDataHO( self, balance = True ):
        
        trainDataPath = self.getDataPathHO( 'train' )
        trainData = iter_loadtxt( trainDataPath )
        
        testDataPath = self.getDataPathHO( 'test' )
        testData = iter_loadtxt( testDataPath )
        
        trainFeature, trainLabel = self.processData( trainData, 'train' )
        if balance == True:
            trainFeature, trainLabel = self.balanceData( trainFeature, trainLabel )
        
        testFeature, testLabel = self.processData( testData, 'test' )
        
        return trainFeature, trainLabel , testFeature, testLabel
    
    # load data that use hold-out test strategy
    def loadDataSingle( self ):
        
        trainDataPath = self.getDataPathSingle(  )
        trainData = iter_loadtxt( trainDataPath )
        
        trainFeature, trainLabel = self.processData( trainData )
        
        return trainFeature, trainLabel
    
    def normalizeFeature( self, feature, setType ):
        
        if setType == 'train':
            self.mean = np.mean( feature )
            self.var = np.var( feature )
            # zero-mean, and unit variance
            normalizedFeature = ( ( feature - np.mean( feature ) ) /math.sqrt( np.var( feature ) ) )
        elif setType == 'test':
            normalizedFeature = ( ( feature - self.mean ) /math.sqrt( self.var ) )
        
        return normalizedFeature
    
    def getDataSize( self ):
        if self.dataType == 'waveform' or self.dataType == 'toyWaveform':
            dataSize = self.audioLength *self.sampleRate
        elif self.dataType == 'spectrogram' or self.dataType == 'toySpectrogram':
            dataSize = self.timeSteps *self.fftBins
        else:
            print( 'DataType Error' )
        return dataSize
    
    def processData( self, data, setType ):
        
        dataSize = self.getDataSize(  )
            
        data = data.astype( 'float32' )
        np.random.seed( seed = 7 )
        np.random.shuffle( data )
        
        # get feature
        feature = data[ :, 0: dataSize ]
        
        # add hamming window for each frame
        feature = self.addHamming( feature, self.timeSteps )
        
        # normalize feature
        feature = self.normalizeFeature( feature, setType )
        
        # get and one-hot-encode the label
        label = data[ :, dataSize + self.labelPosition ]
        label = np_utils.to_categorical( label )
        
        # print how many classes 
        print( 'The number of classes are : ' + str( label.shape[ 1 ] ) )
        
        return feature, label 
    
    def balanceData(self, feature, label ):
        
        # record the dimension of the feature
        featureSize = feature.shape[ 1 ]
        
        # random oversampling
        ros = RandomOverSampler( random_state= 7 )
        feature, label = ros.fit_sample( feature, np.argmax( label, 1 ) )
        numSamples = len( label )
        label = np.array( label )
        label.resize( [ numSamples, 1 ] )
        dataSet = np.concatenate( ( feature, label ), axis = 1 ) 
        np.random.shuffle( dataSet )
        feature = dataSet[ :, 0: featureSize ]
        label = dataSet[ :, featureSize ]
        label = np_utils.to_categorical( label )
        
        return feature, label 
    
    def addHamming( self, inputMatrix, timeStepNum ):
        print( type( inputMatrix ) )
        sequenceLength = int( inputMatrix.shape[ 1 ] )
        subSequenceLength = sequenceLength / timeStepNum
        hammingWindow = np.hamming( subSequenceLength )
        for sampleIndex in range( 0, int( inputMatrix.shape[ 0 ] ) ):
            for timeStep in range( 0, timeStepNum ):
                start = int( timeStep *subSequenceLength )
                end = int( ( timeStep + 1 ) * subSequenceLength )
                inputMatrix[ sampleIndex, start :end ] *= hammingWindow
        return inputMatrix

if __name__ == '__main__':
    database = SpeechDatabase(  )
    trainFeature, trainLabel = database.loadData(  )
    #plotInputDistribution( a, saveFolder = '' )
    
