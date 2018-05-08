import numpy as np

class PitchDatabase( SpeechDatabase ):
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
    
class PitchDatabaseNoNoise( PitchDatabase ):
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_100NoNoise.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20NoNoise.csv'
        return filePath 
    
class PitchDatabaseNoNoiseWideEnergyRange( PitchDatabase ):
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_200NoNoiseWideEnergyRange.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20NoNoiseWideEnergyRange.csv'
        return filePath 
    
class PitchDatabaseNoNoiseFixedEnergy( PitchDatabase ):
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_100NoNoiseFixedEnergy.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20NoNoiseFixedEnergy.csv'
        return filePath 
    
class PitchDatabaseNoNoiseFixedEnergyAndTime( PitchDatabase ):
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_100NoNoiseFixedEnergyAndTime.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20NoNoiseFixedEnergyAndTime.csv'
        return filePath 
    
class PitchDatabaseNoNoiseFixedTime( PitchDatabase ):
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_100NoNoiseFixedTime.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20NoNoiseFixedTime.csv'
        return filePath 
    
class PitchDatabaseFixedTimeRandNoise( PitchDatabase ):
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_800FixedTimeRandNoise.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20FixedTimeRandNoise.csv'
        return filePath 
    
class PitchDatabaseNoNoiseWideEnergyRangeCloseToCenter( PitchDatabase ):
    groundTruth = np.linspace( 3750, 4250, 11 )
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_16_200NoNoiseWideEnergyRangeCTC.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_16_20NoNoiseWideEnergyRangeCTC.csv'
        return filePath 
    
class PitchDatabaseWideEnergyRangeCloseToCenter( PitchDatabase ):
    groundTruth = np.linspace( 3750, 4250, 11 )
    def getDataPathHO( self, targetSet ):
        if self.dataType == 'waveform':
            if targetSet == 'train':
                filePath = 'I:\database\pitch\data\dataset_11_200WideEnergyRangeCTC.csv'
            if targetSet == 'test':
                filePath = 'I:\database\pitch\data\dataset_11_20WideEnergyRangeCTC.csv'
        return filePath 

class PitchSimple( PitchDatabase ):
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