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

   
class PitchTest( SpeechDatabase ):
    # 
    dataType = 'waveform'
    testType = 'holdOut'
    dataName = 'Pitch'
    
    # labelPosition 
    labelPosition = 0
    
    # ground truth frequencies
    groundTruth = np.linspace( 500, 8000, 16 )
    
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_100.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20.csv'
        elif self.dataType == 'toyWaveform' :
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_2.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_1.csv'
        return filePath 
    
class PitchTestNoNoise( PitchTest ):
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_100NoNoise.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20NoNoise.csv'
        return filePath 
    
class PitchTestNoNoiseWideEnergyRange( PitchTest ):
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_200NoNoiseWideEnergyRange.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20NoNoiseWideEnergyRange.csv'
        return filePath 
    
class PitchTestNoNoiseFixedEnergy( PitchTest ):
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_100NoNoiseFixedEnergy.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20NoNoiseFixedEnergy.csv'
        return filePath 
    
class PitchTestNoNoiseFixedEnergyAndTime( PitchTest ):
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_100NoNoiseFixedEnergyAndTime.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20NoNoiseFixedEnergyAndTime.csv'
        return filePath 
    
class PitchTestNoNoiseFixedTime( PitchTest ):
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_100NoNoiseFixedTime.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20NoNoiseFixedTime.csv'
        return filePath 
    
class PitchTestFixedTimeRandNoise( PitchTest ):
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_800FixedTimeRandNoise.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20FixedTimeRandNoise.csv'
        return filePath 
    
class PitchTestNoNoiseWideEnergyRangeCloseToCenter( PitchTest ):
    groundTruth = np.linspace( 3750, 4250, 11 )
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_200NoNoiseWideEnergyRangeCTC.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20NoNoiseWideEnergyRangeCTC.csv'
        return filePath 
    
class PitchTestWideEnergyRangeCloseToCenter( PitchTest ):
    groundTruth = np.linspace( 3750, 4250, 11 )
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_11_200WideEnergyRangeCTC.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_11_20WideEnergyRangeCTC.csv'
        return filePath 

class PitchSimple( PitchTest ):
    groundTruth = [ 900, 950, 1000, 1050, 1100 ]
    audioLength = 1 
    sampleRate = 4000
    timeSteps = 25
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_5_1000simple.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_5_20simple.csv'
        return filePath     
    
# test the frequency response    
class FrequencyTest( SpeechDatabase ):
    # 
    dataType = 'waveform'
    testType = 'singleTest'
    dataName = 'frequencyTest'
    
    # labelPosition 
    labelPosition = 0
    
    # ground truth frequencies
    #groundTruth = np.linspace( 500, 8000, 16 )
    
    def processData( self, data ):
        
        dataSize = self.getDataSize(  )
            
        data = data.astype( 'float32' )
        
        # get feature
        feature = data[ :, 0: dataSize ]
        
        # add hamming window for each frame
        #feature = self.addHamming( feature, self.timeSteps )
        
        # normalize feature
        #feature = self.normalizeFeature( feature, setType )
        
        # get and one-hot-encode the label
        label = data[ :, dataSize + self.labelPosition ]
        #label = np_utils.to_categorical( label )
        
        # print how many classes 
        #print( 'The number of classes are : ' + str( label.shape[ 1 ] ) )
        
        return feature, label 
    
    def getDataPathSingle( self ):
        if self.dataType == 'waveform':
            filePath = 'I:\database\pitch\data\\testSignals.csv'
        return filePath 

# simple, sample rate to 4000, time to 1s 
class FrequencyTestSimple( FrequencyTest ):
    #groundTruth = np.linspace( 10, 2000, 200 )
    audioLength = 1 
    sampleRate = 4000
    timeSteps = 25
    def getDataPathSingle( self ):
        if self.dataType == 'waveform':
            filePath = 'I:\database\pitch\data\\testSignals_simple.csv'
        return filePath 

    
class ToyPitchTest( PitchTest ):
    dataType = 'toyWaveform'

if __name__ == '__main__':
    database = FrequencyTest(  )
    trainFeature, trainLabel = database.loadData(  )
    #plotInputDistribution( a, saveFolder = '' )
    
