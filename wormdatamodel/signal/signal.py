# Authors: Milena Chakraverti-Wuerthwein and Francesco Randi

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import pickle
from scipy.optimize import minimize
from copy import deepcopy as deepcopy
from datetime import datetime
import mistofrutta.struct.irrarray as irrarray
import wormdatamodel as wormdm

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
    which_skip = dict(zip([],[])); # which strides should be ignored
    
    maxY = np.empty((0))
    photobl_p = np.empty((0,0))
    photobl_n_params = 5
    
    logbook = ""
    
    description = "_created from data_"
    filename = "signal.pickle"
    
    def __init__(self, data, info, description = None, 
                 strides = [], stride_names = [], stride_skip = [[0]], 
                 preprocess = None, nan_interp = True, 
                 smooth = False, smooth_n=4, 
                 photobl_calc = False, photobl_appl = False):
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
        stride_names: list of strings (optional)
            See documentation for the irregular array. Default: empty.
        stride_skip: list of integers (optional)
            Number of strides to skip at the beginning. Default: empty.
        preprocess: bool
            Apply the all the preprocessing to the signal. Default: False.        
        '''

        self.data = data;
        self.info = info;
        if description is not None: self.description = description
        
        # Preprocessing
        self.nan_mask = np.isnan(self.data)
        self.maxY = np.zeros(self.data.shape[1])
        self.photobl_p = np.zeros((self.data.shape[1],5))
        
        self.smooth_n = smooth_n; # smoothing parameter
        if preprocess is not None:
            if preprocess:
                nan_interp = True
                smooth = True
                photobl_calc = True
                photobl_appl = True
            elif not preprocess:
                nan_interp = False
                smooth = False
                photobl_calc = False
                photobl_appl = False
        if photobl_appl: photobl_calc = True
        
        if nan_interp: self.data = self.interpolate_nans();
        if photobl_calc: self.calc_photobl()
        if photobl_appl: self.appl_photobl()
        if smooth: self.data = self.smooth(smooth_n);
            
        
        # strides option allows you to construct an irregular array
        self.which_skip = dict(zip(stride_names,stride_skip))
        if len(strides) > 0:
            
            # make sure that stride_names matches in size with the number stride arrays provided
            if len(stride_names) > len(strides):
                warnings.warn('More stride names specified than stride arrays. Unused have been cut.');
                stride_names = stride_names[0:len(strides)]
            elif len(stride_names) < len(strides):
                warnings.warn('Fewer stride names specified than stride arrays. ' + 
                'Additional strides have been set to the default (thing_0, thing_1, ...).');
                for i in np.arange(len(strides) - len(stride_names)):
                    stride_names += ['thing_' + str(i)]
            
            data_uncut = np.copy(self.data)
            self.data = irrarray(data_uncut, strides, strideNames=stride_names)
            
            mask_uncut = np.copy(self.nan_mask)
            self.nan_mask = irrarray(mask_uncut, strides, strideNames=stride_names)
    
    
    @classmethod
    def from_file(cls,folder,filename,*args,**kwargs):
        '''Creates an instance of the class loading the data from file. If 
        filename ends with ".txt" the Signal object will be created from the 
        raw data. If the filename ends with ".pickle", the function will load
        the data from the pickled file. If the filename does not end with either
        extension, the data will be loaded from the pickled file filename.pickle
        if present, otherwise from the raw filename.txt. 
        
        Parameters
        ----------
        folder: str
            Folder containing the file.
        filename: str
            Name of the file containing the signal array.
        *args, **kwargs
            Any other parameter to be passed to the constructor.
        '''
        
        if filename.split(".")[-1] == "txt": 
            # read in data; rows = time, columns = neuron
            data, info = wormdm.signal.from_file(folder,filename)
            # adjusting the shape so that even if only one neuron, still has "columns"
            try:
                data.shape[1]
            except:
                data = np.copy(np.reshape(data,(data.shape[0],1)))
            
            inst = cls(data,info,".".join(filename.split(".")[:-1]),*args,**kwargs)
            return inst
            
        elif filename.split(".")[-1] == "pickle":
            f = open(folder+filename,"rb")
            inst = pickle.load(f)
            f.close()
            return inst
        else:
            if os.path.isfile(folder+filename+".pickle"):
                f = open(folder+filename+".pickle","rb")
                inst = pickle.load(f)
                f.close()
                return inst
            elif os.path.isfile(folder+filename+".txt"):
                data, info = wormdm.signal.from_file(folder,filename)
                try:
                    data.shape[1]
                except:
                    data = np.copy(np.reshape(data,(data.shape[0],1)))
            
                inst = cls(data,info,filename,*args,**kwargs)
                return inst
            else:
                print(folder+filename+" is not present.")
                quit()
                
    @classmethod
    def from_signal_and_reference(cls, folder, uncorr_signal_fname, reference_fname, method=0.0, strides = [], stride_names = [], stride_skip = []):
        
        # If the filename does not have any extension assign one. If both pickle
        # file exist, then load the pickles if the pickle file is newer than
        # the txt, otherwise load the raw txt.
        if len(uncorr_signal_fname.split(".")) == 1:
            if os.path.isfile(folder+uncorr_signal_fname+".pickle") and os.path.isfile(folder+reference_fname+".pickle"):
                if os.path.getmtime(folder+uncorr_signal_fname+".pickle") > os.path.getmtime(folder+uncorr_signal_fname+".txt"):
                    uncorr_signal_fname += ".pickle"
                    reference_fname += ".pickle"
                else:
                    uncorr_signal_fname += ".txt"
                    reference_fname += ".txt"
            else:
                uncorr_signal_fname += ".txt"
                reference_fname += ".txt"
        
        # Extract description and extensions from the filenames
        uncorr_signal_ext = uncorr_signal_fname.split(".")[-1]
        uncorr_signal_descr = uncorr_signal_fname.split(".")[:-1]
        reference_ext = reference_fname.split(".")[-1]
        reference_descr = reference_fname.split(".")[:-1]
        
        # If the two extensions are different, quit.
        if uncorr_signal_ext != reference_ext:
            print("Provide the source of the uncorrected signal and reference from the same file format.")
            quit()
        
        # If loading from the raw txt files, preprocess. If loading from the 
        # pickles, do not preprocess.
        if uncorr_signal_ext == "txt":
            uncorr_signal = cls.from_file(folder, uncorr_signal_fname, nan_interp=True, smooth=False, photobl_calc=True, photobl_appl=False, strides = [], stride_names = [], stride_skip = [])
            reference = cls.from_file(folder, reference_fname, nan_interp=True, smooth=False, photobl_calc=True, photobl_appl=False, strides = [], stride_names = [], stride_skip = [])
            
            # Save the preprocessed files.
            uncorr_signal.to_file(folder,".".join([uncorr_signal_fname.split(".")[0],"pickle"]) )
            reference.to_file(folder,".".join([reference_fname.split(".")[0],"pickle"]))
            
        elif uncorr_signal_ext == "pickle":
            uncorr_signal = cls.from_file(folder, uncorr_signal_fname, preprocess=False)
            reference = cls.from_file(folder, reference_fname, preprocess=False)
            if os.path.isfile(folder+cls.filename):
                signal = cls.from_file(folder,cls.filename)
                if signal.info["correction_method"] == method: return signal
        
        # Merge the infos from the source Signal objects
        info = reference.info
        info["uncorr_signal"] = uncorr_signal.info
        info["reference"] = reference.info 
        info["correction_method"] = method
        
        # Compute the corrected signal based on one of the methods available.
        signal_d = np.zeros_like(reference.data)
        if method == 0.0:
            X = np.arange(uncorr_signal.data.shape[0])
            for k in np.arange(uncorr_signal.data.shape[1]):
                unc_sig_pb = uncorr_signal._double_exp(X,uncorr_signal.photobl_p[k])
                ref_pb = reference._double_exp(X,reference.photobl_p[k])
                signal_d[:,k] = (uncorr_signal.data[:,k])/(reference.data[:,k])*(ref_pb*reference.maxY[k])/(unc_sig_pb*uncorr_signal.maxY[k])
        elif method==1.5:
            X = np.arange(uncorr_signal.data.shape[0])
            for k in np.arange(uncorr_signal.data.shape[1]):
                P = reference.photobl_p
                Y = cls._double_exp(X,P[k])
                prop = (Y[1:]-P[k,-1])/(Y[:-1]-P[k,-1])
                oneplusdelta = (reference[1:,k]/reference[:-1,k])*prop
                oneplusd = uncorr_signal[1:,k]/uncorr_signal[:-1,k]/oneplusdelta
                signal_d[0,k] = uncorr_signal[0,k]
                for i in np.arange(uncorr_signal.data.shape[0]-1):
                    signal_d[i+1,k] = signal_d[i,k]*oneplusd[i]
        elif method == 1.6:
            X = np.arange(uncorr_signal.data.shape[0])
            for k in np.arange(uncorr_signal.data.shape[1]):
                P = reference.photobl_p
                Y = cls._double_exp(X,P[k])
                prop = (Y[1:]-P[k,-1])/(Y[:-1]-P[k,-1])
                whatRshouldbe = (reference[:-1,k]-P[k,-1])*prop+P[k,-1]
                oneplusdelta = reference[1:,k]/whatRshouldbe
                oneplusd = (uncorr_signal[1:,k]+100)/(uncorr_signal[:-1,k]+100)/oneplusdelta
                signal_d[0,k] = uncorr_signal[0,k]+100
                for i in np.arange(uncorr_signal.data.shape[0]-1):
                    signal_d[i+1,k] = signal_d[i,k]*oneplusd[i]
        
        if method == 1.6: photobl_calc = photobl_appl = True
        else: photobl_calc = photobl_appl = False
        #photobl_calc = photobl_appl = False
        
        # Create the Signal object with the corrected signal.
        signal = cls(signal_d, info, description="signal", strides=strides, stride_names=stride_names, stride_skip=stride_skip, photobl_calc=photobl_calc, photobl_appl=photobl_appl)
            
        # Transfer logbooks from the original Signal objects
        signal.logbook += "Uncorrected signal log:\n"+"\t"+"\n\t".join(uncorr_signal.logbook.split("\n"))[:-2]+"\n"
        signal.logbook += "Reference log:\n"+"\t"+"\n\t".join(reference.logbook.split("\n"))[:-2]+"\n"
        
        # Transfer nanmask
        signal.nan_mask = np.logical_or(reference.nan_mask,signal.nan_mask)
            
        signal.to_file(folder,cls.filename)
            
        return signal
        
    def to_file(self,folder,filename):
        if filename.split(".")[-1] != "pickle":
            filename += ".pickle"
        pickle_file = open(folder+filename,"wb")
        pickle.dump(self,pickle_file)
        pickle_file.close()
        
    def log(self, s = None, print_to_terminal = True):
        '''Write an entry in the internal log and, if requested, print the same
        entry to terminal. In the log, the entry will start with the 
        current time.
        
        Parameters
        ----------
        s: string
            Text of the entry.
        print_to_terminal: boolean
            If True, the entry will also be printed to terminal. Default: True.

        Returns
        -------
        None
        '''
        if s is not None:
            now = datetime.now()
            dt = now.strftime("%Y-%m-%d %H:%M:%S: ")
            self.logbook += (dt+s+"\n")
            if print_to_terminal:
                print("Signal "+self.description+": "+s)
    
    ##### Pre-processing Functions #####
    
    def interpolate_nans(self):
        '''Replace nans with an interpolated value.'''
        
        self.log("Replacing nans with the interpolated value.",False)
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
        self.log("Smoothing signal with a window of "+str(n)+" points.",False)
        
        sm = np.ones(n)/n
        smoothed = np.copy(self.data)    
        
        for i in np.arange(self.data.shape[1]):
            smoothed[:,i] = np.convolve(self.data[:,i],sm,mode="same")
        
        return smoothed
        
    @staticmethod    
    def _double_exp(X,P):
        '''Photobleaching correction target function'''
        Y = P[0]*np.exp(-X*np.abs(P[1])) + P[2]*np.exp(-X*np.abs(P[3])) + np.abs(P[-1])
        return Y
        
    @staticmethod    
    def _error(P,f,X,Y):
        '''Photobleaching correction error function'''
        e = np.sum(np.power(f(X,P)-Y,2))
        return e
            
    def calc_photobl(self, j=None):
        data_corr = np.copy(self.data)
        X = np.arange(self.data.shape[0])
        self.log("Calculating photobleaching correction, but not applying it.",True)
        
        if j is None:
            iterate_over = np.arange(self.data.shape[1])
        else:
            try: len(j)
            except: j = [j]
            iterate_over = j 
        
        for k in iterate_over:
            print("\t"+str(np.around(float(k)/self.data.shape[1],4))+" done.   ",end="")
            try:
                self.maxY[k] = np.max(data_corr[:,k])
                Y = data_corr[:,k]/self.maxY[k]
                mask = np.ones_like(Y,dtype=np.bool)
                P = np.array([1.,0.006,1.,0.001,0.2])
                
                it = 0
                while True and it<100:
                    R = minimize(self._error,P,args=(self._double_exp,X[mask],Y[mask]))
                    if np.sum(np.absolute((P-R.x)/P)) < 1e-2: break
                    P = R.x
                    
                    std = np.std(self._double_exp(X[mask],P)-Y[mask])
                    mask[:] = np.absolute(self._double_exp(X,P)-Y) < 2.*std
                    it += 1
                
                self.photobl_p[k] = P    
                print("\r",end="")
            except Exception as e:
                self.log("Problems with trace "+str(k)+": "+str(e))

    def appl_photobl(self, j=None):
        self.log("Applying the photobleaching correction.",True)
        X = np.arange(self.data.shape[0])
        if j is None:
            iterate_over = np.arange(self.data.shape[1])
        else:
            try: len(j)
            except: j = [j]
            iterate_over = j
            
        for k in iterate_over:
            data_photobleach_fit = self._double_exp(X,self.photobl_p[k])*self.maxY[k]
            self.data[:,k] /= (1.+data_photobleach_fit)
            
    #def get_photobl_fit(self, X, k=None):
        
    
    ##### Additional Capabilities #####
    
    def trim(self,stride_name, adjust = None):
        '''If the signal is an irregular array, trim it to make the regularize
        the stride along the irregular axis.
        
        Parameters
        ----------
        stride_name: string
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
            start = self.data.firstIndex[stride_name];
            strideLength = np.diff(start);
        except:
            print('Trim unsuccesful, signal has no strides by the name "' + stride_name + '".')
            quit()
        
        mask = np.ones(strideLength.shape, dtype = bool);
        mask[self.which_skip[stride_name]] = False;
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
        trimmed.data = irrarray(temp_data, [temp_strides], strideNames=[stride_name])
        trimmed.nan_mask = irrarray(temp_nan, [temp_strides], strideNames=[stride_name])
        trimmed.which_skip = dict({stride_name : []});
        
        return trimmed
        
    def average(self,stride_name, adjust = None):
        '''Average the signal over an irregular stride. The function first
        obtains the trimmed version of the array along that stride, subtracts
        the background, and averages across the events.
        
        Parameters
        ----------
        stride_name: str
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
            trimmed = self.trim(stride_name, adjust = adjust);
        except:
            print('Average unsuccessful, signal has no strides by the name "' + stride_name + '".')
            quit()
        length = trimmed.data.firstIndex[stride_name][1];
        numStrides = trimmed.data.firstIndex[stride_name].size-1
        temp = np.reshape(trimmed.data,(numStrides,length,trimmed.data.shape[1])) 
        avg = np.mean(temp,axis = 0)
        return avg
        
    def get_loc_std(self,window=8):
        loc_std = np.zeros(self.data.shape[1])
        for j in np.arange(self.data.shape[1]):
            loc_std[j] = np.sqrt(np.median(np.var(self.rolling_window(self.data[:,j], window), axis=-1)))
            
        return loc_std
    
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
        
        
    @staticmethod
    def rolling_window(a, window):
        pad = np.ones(len(a.shape), dtype=np.int32)
        pad[-1] = window-1
        pad = list(zip(pad, np.zeros(len(a.shape), dtype=np.int32)))
        a = np.pad(a, pad,mode='reflect')
        shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
        strides = a.strides + (a.strides[-1],)
        return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
