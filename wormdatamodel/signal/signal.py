# Authors: Milena Chakraverti-Wuerthwein and Francesco Randi

import numpy as np
import matplotlib.pyplot as plt
import mistofrutta.struct.irrarray as irrarray
import wormdatamodel as wormdm
from copy import deepcopy as deepcopy
import warnings
import sys

class Signal:
    '''Class representing signal extracted from recording. It supports irregular
    striding of the signal, useful for recordings with associated events, like
    optogenetics stimulations. The metadata about the irregular strides can be 
    obtained via the recording.get_events() method.
    
    The class provides some preprocessing functionalities, like 
    nan-interpolation, smoothing, and in the future will provide photobleaching
    correction, with the goal of abstracting all this away from the scripts
    analysing the signal.    
    '''

    nan_mask = np.zeros(1) # where nans
    data = np.empty((0,0)); # data array; either ndarray or irrarray
    info = [];
    whichSkip = dict(zip([],[])); # which strides should be ignored
    
    def __init__(self,data,info,strides = [], strideNames = [], strideSkip = [], preprocess = True, smooth_n=4):
        '''Constructor for the class. 
        
        Parameters
        ----------
        data: numpy array
            2D array containing the data. The outermost axis has to be the one
            that gets strided irregularly.
        info: dict
            Dictionary containing the metadata about how the signal has been
            extracted from the frames.
        strides: list of numpy arrays (optional)
            Usually, the ones returned from recording.get_events().
            See documentation for the irregular array. Default: empty.
        strideNames: list of strings (optional)
            See documentation for the irregular array. Default: empty.
        strideSkip: list of integers (optional)
            Number of strides to skip at the beginning. Default: empty.
        preprocess: bool
            Apply the preprocessing to the signal. Default: True.        
        '''
        
        self.smooth_n = smooth_n; # smoothing parameter
        
        self.data = data;
        self.info = info;
        self.nan_mask = np.isnan(self.data)
        
        if preprocess:
            # interpolate
            self.data = self.interpolate_nans();
            self.data = self.smooth(smooth_n);
        
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
        '''Creates an instance of the class loading the data from file.
        
        Parameters
        ----------
        folder: str
            Folder containing the file.
        filename: str
            Name of the file containing the signal array.
        *args, **kwargs
            Any other parameter to be passed to the constructor.
        '''
            
        # read in data; rows = time, columns = neuron
        data, info = wormdm.signal.from_file(folder,filename)
        # adjusting the shape so that even if only one neuron, still has "columns"
        try:
            data.shape[1]
            
        except:
            data = np.copy(np.reshape(data,(data.shape[0],1)))
        
        return cls(data,info,*args,**kwargs)
    
    ##### Pre-processing Functions #####
    
    def interpolate_nans(self):
        '''Replace nans with an interpolated value.'''
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
        '''Smooth the signal with a rectangular filter.
        
        Parameters
        ----------
        n: int
            Width of the rectangular filter.        
        '''
        sm = np.ones(n)/n
        smoothed = np.copy(self.data)    
        
        for i in np.arange(self.data.shape[1]):
            smoothed[:,i] = np.convolve(self.data[:,i],sm,mode="same")
        
        return smoothed
        
    ##### Additional Capabilities #####
    
    def trim(self,strideName, adjust = None):
        '''If the signal is an irregular array, trim it to make the regularize
        the stride along the irregular axis.
        
        Parameters
        ----------
        strideName: string
            Name of the stride along which to trim.
        adjust: int
            Number of points to average to subtract the background.
            
        Returns
        -------
        trimmed: irregular array
            Irregular array that has now effectively a regular stride. 
            trimmed.data can now be copied and reshaped into a multidimensional
            numpy array.        
        '''
        
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
        '''Average the signal over an irregular stride. The function first
        obtains the trimmed version of the array along that stride, subtracts
        the background, and averages across the events.
        
        Parameters
        ----------
        strideName: str
            Name of the irregular stride.
        adjust: int
            Number of points to average for the background subtraction.
            
        Returns
        -------
        avg: numpy array
            Array containing the average over the specified stride.
        
        '''
        
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
        '''Allow for direct indexing of the class to access the data.'''
        return self.data.__getitem__(i)
        
    def __setitem__(self, i, value):
        '''Allow for direct indexing of the class to write in the data.'''
        self.data.__setitem__(i,value)
        
    def __call__(self, *args, **kwargs):
        '''Upon call, use the __call__ method of the data irrarray.'''
        return self.data.__call__(*args, **kwargs)
