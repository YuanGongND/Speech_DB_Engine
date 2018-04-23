
"""
Created Thur Apr 5 18:06:19 2018

@author: Royce
"""

import os
import numpy as np
import scipy.io.wavfile
from subprocess import call
import sys

class Sonant():

    def __init__(self,phn,wav,speaker,dialect):
        self.phn = phn
        self.wav = wav
        self.spkr = speaker
        self.dlct = dialect

class Timit():
    """ Timit class loads the Timit database into memory 
        and has multiple class methods that operate on the data.
    
    :param example: type, description.
    :param dbPath: str, path to TIMIT database.
    """

    def __init__(self, dbPath, xLen = 10000):
        self.sonant_dataset = {}
        self.xLen = xLen
        self.phnList = []
        self.dlctList = []
        self.spkrList = []
        self.timit = self.load_db(dbPath) 

    def load_db(self,dbPath):
        sys.stdout.write('loading...')       
        trainPath = dbPath + '/TRAIN'
        testPath = dbPath + '/TEST'
        #TODO: test-core
        
        for path in [trainPath, testPath]:
            for subdir, dirs, files in os.walk(path):
                sys.stdout.write('.')
                sys.stdout.flush()
                # reinstantiated here so 2 don't match across speakers
                file_matcher = {}
                for file in files:
                    if file.lower().endswith(('.phn','.wav')):
                        basename = file.split('.')[0]
                        if basename in file_matcher:
                            # parse speaker and dialect info
                            subdir_arr = subdir.split('/')
                            spkr = subdir_arr[-1]
                            if spkr not in self.spkrList:
                                self.spkrList.append(spkr)
                            dlct = subdir_arr[-2]
                            if dlct not in self.dlctList:
                                self.dlctList.append(dlct)
                    
                            # parse wav and phn files
                            basepath = os.path.join(subdir,basename)
                            sonantPairs = self.sonant_tuples(basepath+'.phn',basepath+'.wav',subdir)
                            if len(sonantPairs) == 0:
                                # there was an exception in sonant_tuples
                                pass
                            for pair in sonantPairs:
                                phn_str = pair[0]
                                if phn_str not in self.phnList:
                                   self.phnList.append(phn_str) 
                                wav = pair[1]
                                if phn_str not in self.sonant_dataset:
                                    self.sonant_dataset[phn_str] = [Sonant(phn_str, wav, spkr,dlct)]
                                else: 
                                    self.sonant_dataset[phn_str].append(Sonant(phn_str, wav, spkr,dlct))
                        else:
                            # add it to purgatory
                            file_matcher[basename] = 1

    def sonant_tuples(self, phn_file, wav_file, path):
        ''' returns a list of sonant text/waveform pairs based on the input files
        '''
        # TODO: instead of calling a subprocess might look into sphfile library here:
        # https://pypi.python.org/pypi/sphfile/1.0.0
        # TODO: other alternative:
        # import soundfile as sf
        # data, samplerate = sf.read('existing_file.wav')
        try:
            call(['sph2pipe', '-f', 'wav', wav_file, path+'/temp.WAV'])
            rate, data = scipy.io.wavfile.read(path + '/temp.WAV')
            call(['rm', path + '/temp.WAV'])
        except:
            print('error parsing: ' + wav_file)
            return ''
        pairs = []
        # parse the .PHN file
        for line in open(phn_file):
            phn_list = line.split()
            pairs.append((phn_list[2], data[int(phn_list[0]):int(phn_list[1])]))

        # doulbe check that pairs is on stack and gets recreated on call
        return pairs

    def read_db(self, yType, yVals):
        y = np.zeros(1, int)
        x = np.zeros((1,self.xLen),int)
        first = True
        # returns ['sh',..] or ['DR1',...] or ['FCAO1',...]
        yValList = self.ytype_val_list(yType)
        # converting yVals array to corresponding numbers
        yNumVals = [ yValList.index(y) for y in yVals] 
        print('yNumVals: ' + str(yNumVals))
        count = 1
        # TODO: might need to get parallel
        for sonant in self.sonant_dataset[yValList[0]]:
            print('percent done: ' + str((count/12598)*100))
            count = count + 1 
            # returns 'sh' or 'DR1' or 'FCAO1'
            sonantYVal = self.ytype_value(yType,sonant)
            if sonantYVal in yValList:
                if first:
                    y[0] = yValList.index(sonantYVal)
                    x[0] = self.cut_pad(sonant.wav)
                    first = False
                else:
                    y = np.append(y,yValList.index(sonantYVal))
                    print(sonant.phn)
                    print(str(len(sonant.wav)))
                    print(str(len(self.cut_pad(sonant.wav))))
                    x = np.append(x,[self.cut_pad(sonant.wav)],axis=0)

        return y,x

    def cut_pad(self, wav):
        if len(wav) < self.xLen:
            return np.append(wav, np.zeros(self.xLen - len(wav), int))
        elif len(wav) > self.xLen:
            return wav[:self.xLen]
        else:
            return wav

    # TODO: these are bad come back and condense
    def ytype_value(self, yType, sonant):
        if yType == 'PHN':
            return sonant.phn
        elif yType == 'DLCT':
            return sonant.dialect
        elif yType == 'SPKR':
            return sonant.speaker

    def ytype_val_list(self, yType):
        if yType == 'PHN':
            return self.phnList
        elif yType == 'DLCT':
            return self.dlctList
        elif yType == 'SPKR':
            return self.spkrList

if __name__ == '__main__':
    timit = Timit('/Users/roycebranning/Desktop/Spring18_School/Speech_Research/TIMIT')
    print(timit.sonant_dataset)
    print(len(timit.sonant_dataset['h#']))
    y,x = timit.read_db('PHN',['sh'])
    print('x:' + str(x))
    print('y:' + str(y))
    
