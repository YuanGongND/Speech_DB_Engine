
"""
Created Thur Apr 5 18:06:19 2018

@author: Royce
"""

import os
import numpy as np
import scipy.io.wavfile
from subprocess import call, DEVNULL
import sys

class Sonant():

    def __init__(self,phn,wav,speaker,dialect, dataset):
        self.phn = phn
        self.wav = wav
        self.spkr = speaker
        self.dlct = dialect
        self.dataset = dataset

class Timit():
    """ Timit class loads the Timit database into memory 
        and has multiple class methods that operate on the data.
    
    :param example: type, description.
    :param dbPath: str, path to TIMIT database.
    """

    def __init__(self, dbPath, setPath, xLen = 10000):
        self.sonant_dataset = []
        self.xLen = xLen
        self.phnList = []
        self.dlctList = []
        self.spkrList = []
        self.timit = self.load_db(dbPath)
        self.coreTestSet = self.load_set(setPath)

    def load_db(self,dbPath):
        trainPath = dbPath + '/TRAIN'
        testPath = dbPath + '/TEST'
        test_count = 0
        exceptions = list()
        perc = 0 
        for path in [trainPath, testPath]:
            for subdir, dirs, files in os.walk(path):

                # Loading visual
                sys.stdout.write('loading: ' + str(int((perc/647)*100)) + '%') 
                sys.stdout.write('\r')
                sys.stdout.flush()
                perc = perc + 1
               
                # Add Sonants to sonant_dataset 
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
                            dataset = subdir_arr[-3] 
                    
                            # parse wav and phn files
                            basepath = os.path.join(subdir,basename)
                            sonantPairs, err = self.sonant_tuples(basepath+'.phn',basepath+'.wav',subdir)
                            if len(err) != 0:
                                exceptions.append(err)
                            
                            for pair in sonantPairs:
                                phn_str = pair[0]
                                if phn_str not in self.phnList:
                                   self.phnList.append(phn_str) 
                                wav = pair[1]
                                test_count = test_count + 1
                                self.sonant_dataset.append(Sonant(phn_str,wav,spkr,dlct, dataset))
                        else:
                            # add it to purgatory
                            file_matcher[basename] = 1
        
        if len(exceptions) > 0:
            print('\n')
            print('error parsing file(s):')
            for ex in exceptions:
                print(ex)
        print('')

    def load_set(self,setPath):
        set_dict = {}
        with open(setPath,'r') as setFile:
            for row in setFile:
                spkrs = row.strip('\n').split(',')
                key = 'DR' + str(spkrs[0])
                set_dict[key] = [ 'M' + spkrs[1],
                                    'M' + spkrs[2], 'F' + spkrs[3].rstrip() ]

        return set_dict 

    def sonant_tuples(self, phn_file, wav_file, path):
        ''' returns a list of sonant text/waveform pairs based on the input files
        '''
        exception = ''
        try:
            call(['sph2pipe', '-f', 'wav', wav_file, path+'/temp.WAV'], stdout=DEVNULL,stderr=DEVNULL)

            rate, data = scipy.io.wavfile.read(path + '/temp.WAV')
            call(['rm', path + '/temp.WAV'])
        except:
            exception = wav_file
            return '',exception
        pairs = []
        # parse the .PHN file
        for line in open(phn_file):
            phn_list = line.split()
            pairs.append((phn_list[2], data[int(phn_list[0]):int(phn_list[1])]))

        # double check that pairs is on stack and gets recreated on call
        return pairs,exception

    def read_db(self, yType='PHN', yVals='All', dataset='test'):

        # initializing np arrays as zeros bc insertion is faster than appending
        y = np.zeros(300000, int)
        x = np.zeros((300000,self.xLen),int)

        # returns ['sh',..] or ['DR1',...] or ['FCAO1',...] depending on input
        # this should also check that the second parameter is matched right?
        # so 'PHN' and 'sh'
       
        yValList = self.ytype_val_list(yType)
       
        # check the 2nd input for assigning yVals 
        if type(yVals) == str and yVals.lower() == 'all':
            yVals = yValList 

        # converting yValList array to corresponding numbers
        yNumVals = [ yValList.index(i) for i in yVals] 
       
        # counter used for insertion into y and x np arrays 
        count = 0

        for sonant in self.sonant_dataset:
           
            if not self.in_dataset(sonant, dataset):
                continue 

            # checks if value is desired given parameters 
            sonantYVal = self.ytype_value(yType,sonant)
            if yValList.index(sonantYVal) in yNumVals:
                y[count] = yValList.index(sonantYVal)
                x[count] = self.cut_pad(sonant.wav) 
                count = count + 1

        # cut off the extra part 
        y = y[0:count]
        x = x[0:count]
        
        return y,x

    def in_dataset(self, sonant, dataset):
        if dataset.lower() == 'coretest':
            if sonant.spkr in self.coreTestSet[sonant.dlct]:
                return True
        elif dataset.lower() == sonant.dataset.lower():
            return True

        return False

    def cut_pad(self, wav):
        if len(wav) < self.xLen:
            return np.append(wav, np.zeros(self.xLen - len(wav), int))
        elif len(wav) > self.xLen:
            return wav[:self.xLen]
        else:
            return wav

    def ytype_value(self, yType, sonant):
        if yType == 'PHN':
            return sonant.phn
        elif yType == 'DLCT':
            return sonant.dlct
        elif yType == 'SPKR':
            return sonant.spkr

    def ytype_val_list(self, yType):
        if yType == 'PHN':
            return self.phnList
        elif yType == 'DLCT':
            return self.dlctList
        elif yType == 'SPKR':
            return self.spkrList

if __name__ == '__main__':
    timit = Timit('/Users/roycebranning/Desktop/Spring18_School/Speech_Research/TIMIT')
    #print('total len (should be same and count number: ' + str(len))
    y,x = timit.read_db('PHN',['sh','h#'])
    print('x:' + str(x))
    print('y:' + str(y))
    
