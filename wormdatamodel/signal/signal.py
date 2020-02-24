import numpy as np
import matplotlib.pyplot as plt
import mistofrutta.struct.irrarray as irrarray
import wormdatamodel as wormdm
import wormbrain as wormb
from copy import deepcopy as deepcopy
import warnings
import sys

class Signal:

    nan_mask = np.zeros(1) # where nans
    data = np.empty((0,0)); # data array; either ndarray or irrarray
    info = [];
    whichSkip = dict(zip([],[])); # which strides should be ignored
    
    def __init__(self,data,info,strides = [], strideNames = [], strideSkip = [], preprocess = True):
        
        NS = 4; # smoothing parameter
        
        self.data = data;
        self.info = info;
        self.nan_mask = np.isnan(self.data)
        
        if preprocess:
            # interpolate
            self.data = self.interpolate_nans();
            self.data = self.smooth(NS);
        
        # strides option allows you to construct an irregular array
        self.whichSkip = dict(zip(strideNames,strideSkip))
        if len(strides) > 0:
            
            # make sure that strideNames matches in size with the number stride arrays provided
            if len(strideNames) > len(strides):
                warnings.warn('More stride names specified than stride arrays. Unused have been cut.');
                strideNames = strideNames[0:len(strides)]
            elif len(strideNames) < len(strides):
                warnings.warn('Fewer stride names specified than stride arrays. ' + 
                'Additional strides have been set to the default (thing_0, thing_1, ...).');
                for i in np.arange(len(strides) - len(strideNames)):
                    strideNames += ['thing_' + str(i)]
            
            data_uncut = np.copy(self.data)
            self.data = irrarray(data_uncut, strides, strideNames=strideNames)
            
            mask_uncut = np.copy(self.nan_mask)
            self.nan_mask = irrarray(mask_uncut, strides, strideNames=strideNames)
    
    
    @classmethod
    def from_file(cls,folder,filename,*args,**kwargs):
        # read in data; rows = time, columns = neuron
        data, info = wormdm.signal.from_file(folder,filename)
        
        return cls(data,info,*args,**kwargs)
    
    ##### Pre-processing Functions #####
    
    def interpolate_nans(self):
        # replace nans with an interpolated value
        interpolated = np.copy(self.data)
        
        for i in np.arange(self.data.shape[1]):
            # nans: location of nans
            # x: function that finds the non-zero entries
            nans, x = self.nan_mask[:,i], lambda z: z.nonzero()[0]
            try:
                interpolated[nans,i] = np.interp(x(nans), x(~nans), self.data[~nans,i])
            except:
                pass
        
        return interpolated
    
    def smooth(self, n):
        sm = np.ones(n)/n
        smoothed = np.copy(self.data)    
        
        for i in np.arange(self.data.shape[1]):
            smoothed[:,i] = np.convolve(self.data[:,i],sm,mode="same")
        
        return smoothed
        
    ##### Additional Capabilities #####
    
    def trim(self,strideName, adjust = None):
        try:
            start = self.data.firstIndex[strideName];
            strideLength = np.diff(start);
        except:
            print('Trim unsuccesful, signal has no strides by the name "' + strideName + '".')
            sys.exit();
        
        mask = np.ones(strideLength.shape, dtype = bool);
        mask[self.whichSkip[strideName]] = False;
        min_len = np.min(strideLength[mask])
        
        temp_data = np.ones((1,self.data.shape[1]))
        temp_nan = np.ones((1,self.data.shape[1]))
        
        for stPt in start[np.append(mask,False)]:
            if adjust == None: adj = 0;
            else: adj = np.mean(self.data[stPt:stPt+adjust],axis=0);
            temp_data = np.vstack((temp_data,self.data[stPt:stPt+min_len]-adj));
            temp_nan = np.vstack((temp_nan,self.nan_mask[stPt:stPt+min_len]));
        temp_data = np.copy(temp_data[1:])
        temp_nan = np.copy(temp_nan[1:])
        
        temp_strides = (np.ones_like(strideLength[mask])*min_len)
        #temp_strides = (np.ones_like(strideLength)*min_len)
        # print('trimmed strides',temp_strides)
        
        trimmed = self.copy()
        trimmed.data = irrarray(temp_data, [temp_strides], strideNames=[strideName])
        trimmed.nan_mask = irrarray(temp_nan, [temp_strides], strideNames=[strideName])
        trimmed.whichSkip = dict({strideName : []});
        
        return trimmed
        
    def average(self,strideName, adjust = None):
        # adjust tells you how many points to average as a baseline to subtract out
        try:
            trimmed = self.trim(strideName, adjust = adjust);
        except:
            print('Average unsuccessful, signal has no strides by the name "' + strideName + '".')
            sys.exit();
        length = trimmed.data.firstIndex[strideName][1];
        numStrides = trimmed.data.firstIndex[strideName].size-1
        temp = np.reshape(trimmed.data,(numStrides,length,trimmed.data.shape[1])) 
        avg = np.mean(temp,axis = 0)
        return avg
    
    ##### Underbelly Functions #####
    
    def copy(self):
        return deepcopy(self)
        
    def __getitem__(self, i):
        '''
        Allow for direct indexing of the class to access the data.
        '''
        return self.data.__getitem__(i)
        
    def __setitem__(self, i, value):
        '''
        Allow for direct indexing of the class to write in the data.
        '''
        self.data.__setitem__(i,value)
        
    def __call__(self, *args, **kwargs):
        '''
        Upon call, use the __call__ method of the data irrarray.
        '''
        return self.data.__call__(*args, **kwargs)
