# Authors: Milena Chakraverti-Wuerthwein and Francesco Randi

import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
import pickle
from scipy.optimize import minimize
from scipy.ndimage import median_filter
from scipy.special import comb
from scipy.signal import savgol_coeffs
from copy import deepcopy as deepcopy
from datetime import datetime
import mistofrutta.struct.irrarray as irrarray
import wormdatamodel as wormdm
import savitzkygolay as sg
from sklearn.decomposition import FastICA

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
    
    loc_std_cached = None
    
    logbook = ""
    
    description = "_created from data_"
    filename = "signal.pickle"
    
    matchless_nan_th_fname = "matchless_nan_th.txt"
    
    def __init__(self, data, info, description = None, 
                 strides = [], stride_names = [], stride_skip = [[0]], 
                 preprocess = None, nan_interp = True, inf_remove = True,
                 smooth = False, smooth_n=3, smooth_mode="rectangular", 
                 smooth_poly=1, remove_spikes = False,
                 photobl_calc = False, photobl_appl = False,
                 corr_inst_photobl = False):
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
            Apply the all the preprocessing to the signal. It does not control 
            the correction of instantaneous photobleaching. Default: False.     
        nan_interp: bool.
            Interpolate nans. Default: True.
        smooth: bool
            Smooth with a box of size smooth_n. Default: False.
        smooth_n: int (optional)
            Size of the smoothing box. Default: 3.
        smooth_mode: str (optional)
            Type of smoothing (rectangular or sg). Default: rectangular.    
        smooth_poly: int (optional)
            Polynomial order of the Savitzky-Golay smoothing. Default: 1.
        photobl_calc: bool
            Calculate the photobleaching correction. Set this to True instead of
            photobl_appl if you are using the photobleaching correction for one
            of the ratiometric methods. Default: False.
        photobl_appl: bool
            Apply the photobleaching correction. If True, the photobleaching 
            correction will be calculated. Default: False.
        corr_inst_photobl: bool
            Detect and correct instantaneous photobleaching. Use only for the
            red signal. Default: False.
        '''

        self.data = data;
        self.info = info;
        if description is not None: self.description = description
        
        # Preprocessing
        self.nan_mask = np.isnan(self.data)
        self.maxY = np.zeros(self.data.shape[1])
        self.photobl_p = np.zeros((self.data.shape[1],5))
        
        # Preprocessing flags
        self.nan_interpolated = False
        self.inf_removed = False
        self.inst_photobl_corrected = False
        self.photobl_calculated = False
        self.photobl_applied = False
        self.spikes_removed = False
        self.smoothed = False
        
        self.spikes = np.zeros_like(data,dtype=bool)
        
        self.apply_preprocessing(
         preprocess=preprocess, nan_interp=nan_interp, inf_remove = inf_remove,
         smooth = smooth, smooth_n=smooth_n, smooth_mode=smooth_mode, 
         smooth_poly=smooth_poly, remove_spikes = remove_spikes,
         photobl_calc = photobl_calc, photobl_appl = photobl_appl,
         corr_inst_photobl = corr_inst_photobl)
        
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
        
        if smooth_n>3 and smooth_poly<5 and smooth_poly<(smooth_n-1):
            self.derivative = self.get_derivative(self.data,smooth_n,smooth_poly)
        else:
            self.derivative = self.get_derivative(self.data,3,1)
        
    def apply_preprocessing(
                 self, preprocess = None, nan_interp = True, 
                 inf_remove = True,
                 smooth = False, smooth_n=3, smooth_mode="rectangular", 
                 smooth_poly=1, remove_spikes = False,
                 photobl_calc = False, photobl_appl = False,
                 corr_inst_photobl = False):
    
        self.smooth_n = smooth_n; # smoothing parameter
        if preprocess is not None:
            if preprocess:
                nan_interp = True
                inf_remove = True
                smooth = True
                remove_spikes = True
                photobl_calc = True
                photobl_appl = True
            elif not preprocess:
                nan_interp = False
                inf_remove = False
                smooth = False
                remove_spikes = False
                photobl_calc = False
                photobl_appl = False
        if photobl_appl: photobl_calc = True
                
        if nan_interp and not self.nan_interpolated: self.interpolate_nans()
        if inf_remove and not self.inf_removed: self.remove_infs()
        if corr_inst_photobl and not self.inst_photobl_corrected: self.corr_inst_photobl()
        if photobl_calc and not self.photobl_calculated: self.calc_photobl()
        if photobl_appl and not self.photobl_applied: self.appl_photobl()
        if remove_spikes and not self.spikes_removed: self.remove_spikes()
        if smooth and not self.smoothed: self.smooth(smooth_n,None,smooth_poly,smooth_mode)
        if inf_remove: self.remove_infs() # Yes, again
    
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
        
        appl_preproc = False
        from_pickle = False
        nan_th = None
        
        if "matchless_nan_th_from_file" in kwargs.keys():
            nan_th_ = cls.get_matchless_nan_th_from_file(folder)
            if nan_th_ is not None:
                nan_th = nan_th_
                print("Signal: using matchless_nan_th_from_file=",nan_th)
            kwargs.pop("matchless_nan_th_from_file")
        
        if "matchless_nan_th" in kwargs.keys():
            if kwargs["matchless_nan_th"] is not None:
                nan_th = kwargs["matchless_nan_th"]
            kwargs.pop("matchless_nan_th")
        
        if filename.split(".")[-1] == "txt": 
            fname_base = ".".join(filename.split(".")[:-1])
            # read in data; rows = time, columns = neuron
            data, info = wormdm.signal.from_file(folder,filename)
            # adjusting the shape so that even if only one neuron, still has "columns"
            try:
                data.shape[1]
            except:
                data = np.copy(np.reshape(data,(data.shape[0],1)))
                
            inst = cls(data,info,".".join(filename.split(".")[:-1]),*args,**kwargs)
            #return inst
            
        elif filename.split(".")[-1] == "pickle":
            fname_base = ".".join(filename.split(".")[:-1])
            f = open(folder+filename,"rb")
            inst = pickle.load(f)
            from_pickle = True
            appl_preproc = True
            f.close()
            #return inst
        else:
            fname_base = filename
            if os.path.isfile(folder+filename+".pickle"):
                f = open(folder+filename+".pickle","rb")
                inst = pickle.load(f)
                f.close()
                appl_preproc = True
                from_pickle = True
                #inst.apply_preprocessing(**kwargs)
                #return inst
            elif os.path.isfile(folder+filename+".txt"):
                data, info = wormdm.signal.from_file(folder,filename)
                try:
                    data.shape[1]
                except:
                    data = np.copy(np.reshape(data,(data.shape[0],1)))
            
                inst = cls(data,info,filename,*args,**kwargs)
                #return inst
            else:
                print(folder+filename+" is not present.")
                quit()
                
        # Substitute data for neurons with too many nans with the matchless
        # signal, if present.
        
        mtchlss_fname = fname_base+"_matchless.txt"
        matchless_present=os.path.isfile(folder+mtchlss_fname)
        if nan_th is not None and matchless_present:
            too_many_nans=np.sum(inst.nan_mask,axis=0)>=nan_th*inst.data.shape[0]
            #print(np.where(too_many_nans))
            
            if np.any(too_many_nans):
                print("Signal "+fname_base+": substituting some with matchless.")
                mtchlss, _ = wormdm.signal.from_file(folder,mtchlss_fname)
                inst.data[:,too_many_nans] = mtchlss[:,too_many_nans]
                inst.nan_mask[:,too_many_nans] = np.isnan(inst.data[:,too_many_nans])
                for i in np.where(too_many_nans)[0]:
                    # nans: location of nans
                    # x: function that finds the non-zero entries
                    nans, x = inst.nan_mask[:,i], lambda z: z.nonzero()[0]
                    try:
                        inst.data[nans,i] = np.interp(x(nans), x(~nans), inst.data[~nans,i])
                    except:
                        pass
                
        if appl_preproc:
            inst.apply_preprocessing(**kwargs)
        
        return inst
                
    @classmethod
    def from_signal_and_reference(
            cls, folder, uncorr_signal_fname="green", reference_fname="red", 
            method=0.0, strides = [], stride_names = [], stride_skip = [], 
            **kwargs):
            
        if folder[-1]!="/": folder+="/"
        
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
            uncorr_signal = cls.from_file(
                             folder, uncorr_signal_fname, 
                             nan_interp=True,remove_spikes=False,smooth=False, 
                             photobl_calc=True, photobl_appl=False, 
                             strides=[], stride_names=[], stride_skip=[])
            reference = cls.from_file(
                             folder, reference_fname, 
                             nan_interp=True,remove_spikes=False,smooth=False, 
                             photobl_calc=True,photobl_appl=False, 
                             corr_inst_photobl=False, 
                             strides = [], stride_names = [], stride_skip = [])
            
            # Save the preprocessed files.
            uncorr_signal.to_file(folder,".".join([uncorr_signal_fname.split(".")[0],"pickle"]) )
            reference.to_file(folder,".".join([reference_fname.split(".")[0],"pickle"]))
            
        elif uncorr_signal_ext == "pickle":
            uncorr_signal = cls.from_file(folder, 
                                          uncorr_signal_fname,preprocess=False)
            reference = cls.from_file(folder, reference_fname,preprocess=False)
            if os.path.isfile(folder+cls.filename):
                signal = cls.from_file(folder,cls.filename,**kwargs)
                if signal.info["correction_method"] == method: 
                    print("Using pickled signal.")
                    return signal
        
        # Merge the infos from the source Signal objects
        info = reference.info.copy()
        info["uncorr_signal"] = uncorr_signal.info
        info["reference"] = reference.info 
        info["correction_method"] = method
        
        # Compute the corrected signal based on one of the methods available.
        signal_d = np.zeros_like(reference.data)
        if method == 0.0:
            # Standard ratiometric
            X = np.arange(uncorr_signal.data.shape[0])
            for k in np.arange(uncorr_signal.data.shape[1]):
                unc_sig_pb = uncorr_signal._double_exp(X,uncorr_signal.photobl_p[k])
                ref_pb = reference._double_exp(X,reference.photobl_p[k])
                #signal_d[:,k] = (uncorr_signal.data[:,k])/(reference.data[:,k])*(ref_pb*reference.maxY[k])/(unc_sig_pb*uncorr_signal.maxY[k])
                # Removing the normalizations, I want it to have the same 
                # amplitude as the raw GCaMP fluorescence, so that it's kind of 
                # not cell-specific.
                signal_d[:,k] = (uncorr_signal.data[:,k])/(reference.data[:,k])*(ref_pb*reference.maxY[k])/(unc_sig_pb)
        elif method==1.5:
            # Derivative-based, version 1.5
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
            # Derivative-based, verions 1.6
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
        elif method == 2.0:
            # ICA
            X = np.arange(uncorr_signal.data.shape[0])
            rlocstd = reference.get_loc_std(8)
            unclocstd = uncorr_signal.get_loc_std()
            trasformer = FastICA(n_components=2, random_state=0)
            for k in np.arange(uncorr_signal.data.shape[1]):
                unc_sig_pb = uncorr_signal._double_exp(X,uncorr_signal.photobl_p[k])
                ref_pb = reference._double_exp(X,reference.photobl_p[k])
                mixed = np.array([uncorr_signal.data[:,k]/unc_sig_pb/unclocstd[k],reference.data[:,k]/ref_pb/rlocstd[k]]).T
                signal_d[:,k] = mixed[:,0]
        elif method == 2.1:
            # Linear subtraction
            pass
        
        
        if method == 1.6: photobl_calc = photobl_appl = True
        else: photobl_calc = photobl_appl = False
        #photobl_calc = photobl_appl = False
        
        # Create the Signal object with the corrected signal.
        signal = cls(signal_d, info, description="signal", strides=strides, 
                     stride_names=stride_names, stride_skip=stride_skip, 
                     photobl_calc=photobl_calc, photobl_appl=photobl_appl,
                     **kwargs)
            
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
        self.nan_interpolated = True
        self.data = interpolated
        
    def remove_infs(self,i=None):
        if i is not None: 
            try: len(i); iterate_over=i
            except: iterate_over = [i]
        else: iterate_over = np.arange(self.data.shape[1])
        
        for j in iterate_over:
            infs = np.where(np.isinf(self.data[:,j]))
            for inf_ in infs:
                self.data[inf_,j] = self.data[inf_-1,j]
                
        self.inf_removed = True
        
    def smooth(self,*args,**kwargs):
        self.unsmoothed_data = np.copy(self.data)
        self.data = self.get_smoothed(*args,**kwargs)
        self.smoothed = True
    
    def get_smoothed(self, n, i=None, poly=1, mode="rectangular"):
        '''Smooth the signal with a rectangular filter.
        
        Parameters
        ----------
        n: int
            Size of the filter.
        i: int (optional)
            Index of the neuron to be smoothed. If None, all the neurons
            are smoothed. Default: None.
        mode: string (optional)
            Smoothing mode. rectangular, sg. Default: rectangular.
        poly: int (optional)
            Polynomial order for the Savitzky Golay filter. Required if mode is
            sg.
        '''
        if mode == "rectangular":
            self.log("Smoothing signal with a window of "+str(n)+" points.",False)
            sm = np.ones(n)/n
            shift = None
        elif mode == "sg":
            self.log("Smoothing signal with a sg filter of size "+str(n)+" and.\
                     order "+str(poly)+".",False)
            #sm = sg.get_1D_filter(n,poly,0)
            sm = savgol_coeffs(n,poly)
            shift = None
        elif mode == "sg_causal":
            self.log("Smoothing signal with a causal sg filter of size "+\
                     str(n)+" and order "+str(poly)+".",False)
            #sm = self.get_causal_sg(n,poly)
            sm = savgol_coeffs(n,poly,pos=n-1)
            shift = (n-1)//2
            
        if i is None:
            smoothed = np.copy(self.data)
            for i in np.arange(self.data.shape[1]):
                if shift is not None:
                    smoothed[shift:,i] = np.convolve(self.data[:,i],
                                                      sm,mode="same")[:-shift]
                else:
                    smoothed[:,i] = np.convolve(self.data[:,i],sm,mode="same")
        else:
            if shift is not None:
                smoothed = np.copy(self.data[:,i])
                smoothed[shift:] = np.convolve(self.data[:,i],
                                                sm,mode="same")[:-shift]
            else:
                smoothed = np.convolve(self.data[:,i],sm,mode="same")
        
        self.smoothed = True
        return smoothed
        
    def remove_spikes(self, i=None):
        if "spikes_removed" not in dir(self): self.spikes_removed=False
        if not self.spikes_removed:
            # For back-compatibility with old objects
            try: self.spikes
            except: self.spikes = np.zeros_like(self.data,dtype=bool)
                
            if i is not None: 
                try: len(i); iterate_over=i
                except: iterate_over = [i]
                self.spikes_removed = True
            else: iterate_over = np.arange(self.data.shape[1])
            
            for j in iterate_over:
                tot_std = np.nanstd(self.data[:,j])
                spikes = np.where(self.data[:,j]-np.average(self.data[:,j])>tot_std*5)[0]
                self.spikes[:,j] = self.data[:,j]-np.average(self.data[:,j])>tot_std*5
                for spike in spikes:
                    self.data[spike,j] = self.data[spike-1,j]
            affected_neurons = "neuron "+str(i) if i is not None else "all neurons"
            self.log("Removing spikes wrt global stdev on "+affected_neurons,False)
        else:
            print("Spikes have already been removed. Not doing it again.",
                  "Start from the unprocessed data.")
                  
    def median_filter(self, i=None):
        if i is not None: 
            try: len(i); iterate_over=i
            except: iterate_over = [i]
        else: iterate_over = np.arange(self.data.shape[1])
        
        for j in iterate_over:
            median_filter(self.data[:,j],3,output=self.data[:,j])
        
        affected_neurons = "neuron "+str(i) if i is not None else "all neurons"
        self.log("Median filtering on "+affected_neurons,False)
                  
    def get_derivative(self,data,n,poly):
        deriv = np.zeros_like(data)
        derker = sg.get_1D_filter(n,poly,1)
        #derker = savgol_coeffs(n,poly,deriv=1)
        n_neurons = data.shape[1]
        for j in np.arange(n_neurons):
            deriv[:,j] = np.convolve(data[:,j],derker,mode="same")
            #for tempo in np.arange(ratio.data.shape[0]):
                #dr[tempo,j] = bdf(ratio.data[:,j],tempo,1)
        return deriv
            
    def get_segment(self,i0,i1,delta=0,
                    baseline=True,baseline_range=None,baseline_mode="constant",
                    normalize="loc_std_restricted",norm_range=None,
                    norm_window=4,unsmoothed_data=False):
        '''Return a pre-processed segment of the data. If baseline is True, the
        function calculates the average of the data in the first delta time
        points and subtract this baseline from the data. If normalize is set 
        to an implemented method, the data will be normalized accordingly.
        
        Parameters
        ----------
        i0: int
            Starting index along the time axis.
        i1: int
            Ending index along the time axis.
        delta: int (optional)
            Number of time points to average to find the baseline. (Default: 0)
        baseline: bool (optional)
            Whether to subtract the baseline. (Default: True)
        normalize: str (optional)
            Normalization method. If it's \"loc_std_restricted\" the data is 
            normalized by the standard deviation of the data from i0 to 
            i0+delta. (Default: loc_std_restricted)
        '''
        if unsmoothed_data and self.smoothed:
            out = self.unsmoothed_data[i0:i1].copy()
        else:
            out = self.data[i0:i1].copy()
        
        # Subtract baseline
        if baseline_range is None: bi0, bi1 = None, delta
        else:
            bi0 = None if baseline_range[0] is None else baseline_range[0]
            bi1 = delta if baseline_range[1] is None else baseline_range[1]
        baseline_s = np.median(out[bi0:bi1],axis=0)
        if baseline and baseline_mode=="constant": 
            out.data -= baseline_s
        elif baseline and baseline_mode=="exp":
            shift = np.min(out[bi0:bi1],axis=0)
            exp_fit_y = out[bi0:bi1]-shift+1e-3
            exp_fit_x = np.arange(len(exp_fit_y))
            for i_neu in np.arange(out.shape[1]):
                a,b = self.lst_sq_exp(exp_fit_x,exp_fit_y[:,i_neu])
                if b>=0: 
                    out[:,i_neu] -= baseline_s[i_neu]
                else:
                    exp_fit_x2 = np.arange(out.shape[0])
                    out[:,i_neu] -= a*np.exp(b*exp_fit_x2)+shift[i_neu]-1e-3
            
        
        # Normalize
        if normalize=="glob_std_restricted":
            # Calculate the restricted global standard deviation
            glob_std = np.nanstd(out[:delta],axis=0)
            # Replace bad loc_std_s with 1, so that the normalization is skipped
            # for those cases.
            msk = (glob_std!=0)*(~np.isnan(glob_std))*(~np.isinf(glob_std))
            glob_std[~msk] = 1.
            out.data /= glob_std
            
        elif normalize=="loc_std_restricted":
            # Cacluate the restricted local standard deviation
            loc_std = self.get_loc_std(out,norm_window)
            # As for previous case
            msk = (loc_std!=0)*(~np.isnan(loc_std))*(~np.isinf(loc_std))
            loc_std[~msk] = 1.
            out.data /= loc_std
            
        elif normalize in ["max","max_abs"]:
            if norm_range is None: mi0, mi1 = delta, None
            else:
                mi0 = delta if norm_range[0] is None else norm_range[0]
                mi1 = None if norm_range[1] is None else norm_range[1]
            if normalize=="max_abs":
                maxs = np.nanmax(np.abs(out[mi0:mi1]),axis=0)
            else:
                maxs = np.nanmax(out[mi0:mi1],axis=0)
            msk = (maxs!=0)*(~np.isnan(maxs))*(~np.isinf(maxs))
            maxs[~msk] = 1.
            out.data /= maxs
        
        return out
        
    def get_segment_nan_mask(self,i0,i1):
        return self.nan_mask[i0:i1,:]
        
    def get_segment_derivative(self,i0,i1):
        return self.derivative[i0:i1,:]
        
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
            
    def calc_photobl(self, j=None, verbose=True):
        data_corr = np.copy(self.data)
        X = np.arange(self.data.shape[0])
        self.log("Calculating photobleaching correction, but not applying it.",verbose)
        
        if j is None:
            iterate_over = np.arange(self.data.shape[1])
            self.photobl_calculated = True
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

    def appl_photobl(self, j=None, verbose=True):
        '''Apply the precomputed photobleaching correction.
        
        Parameters
        ----------
        j: int (optional)
            Neuron to which to apply the correction. If None, the correction 
            will be applied to all the neurons. Default: None.
        ''' 
        
        self.log("Applying the photobleaching correction.",verbose)
        X = np.arange(self.data.shape[0])
        if j is None:
            iterate_over = np.arange(self.data.shape[1])
            self.photobl_applied = True
        else:
            try: len(j)
            except: j = [j]
            iterate_over = j
            
        for k in iterate_over:
            data_photobleach_fit = self._double_exp(X,self.photobl_p[k])*self.maxY[k]
            self.data[:,k] /= (1.+data_photobleach_fit)
            self.data[:,k] *= self.maxY[k]
            
    #def get_photobl_fit(self, X, k=None):
    
    def corr_inst_photobl(self, j=None, poly_width=111, photobl_duration=3, min_distance=30):

        if j is None:
            iterate_over = np.arange(self.data.shape[1])
            self.inst_photobl_corrected = True
        else:
            try: len(j)
            except: j = [j]
            iterate_over = j
            
        # Calculate derivative (and diff, will be useful later)
        deriv = np.zeros((self.data.shape[0],len(iterate_over)))
        diff = np.zeros((self.data.shape[0],len(iterate_over)))
        derker = -sg.get_1D_filter(poly_width,3,1)        
        for i in np.arange(len(iterate_over)):
            k = iterate_over[i]
            deriv[:,i] = np.convolve(derker,self.data[:,k],mode="same")
            diff[:-1,i] = np.diff(self.data[:,k])
            
        # Find where the derivative departs from normal behavior    
        medderiv = np.median(deriv[poly_width:-poly_width],axis=0)
        stdderiv = np.std(deriv[poly_width:-poly_width],axis=0)
        
        for i in np.arange(len(iterate_over)):
            k = iterate_over[i]
            i_pb = np.where(deriv[poly_width:-poly_width,i]<medderiv[i]-3*stdderiv[i])[0]+poly_width
            
            prev_jump_pos = -100*min_distance
            if len(i_pb)>0:
                # Find contiguous regions where the derivative is too negative
                splt = np.where(np.diff(i_pb)>1)[0]+1
                splt = np.append(0,splt)
                splt = np.append(splt,-1)
                for q in np.arange(len(splt)-1):
                    # Find the "real" center of the jump 
                    idx0 = i_pb[splt[q]]
                    idx1 = i_pb[splt[q+1]]
                    if idx1>(idx0+1):
                        #jump_pos_poly = np.argmin(deriv[idx0:idx1,i])+idx0
                        jump_pos = np.argmin(diff[idx0:idx1,i])+idx0
                    else:
                        jump_pos = idx0
                    
                    # Sometimes a region that should be contiguous is split.
                    # If two detected jumps are too close they are likely
                    # originating from this situation, so don't double count
                    # them.
                    if jump_pos<(prev_jump_pos+min_distance): continue
                    prev_jump_pos = jump_pos
                    
                    # Calculate the amplitude of the jump
                    jump = np.sum(deriv[jump_pos-poly_width//2:jump_pos+poly_width//2,i])
                    
                    # Calculate the factor by which the post-jump data needs to 
                    # be multiplied to be corrected
                    # Pre-jump value: could be something more fancy since you're
                    # doing polynomial interpolation
                    pre = np.median(self.data[jump_pos-10:jump_pos,k])
                    mult = pre/(pre+jump)
                    
                    self.data[jump_pos+photobl_duration:,k] *= mult
    
    @staticmethod
    def lst_sq_exp(x,y):
        '''Returns the least-squares fit of an exponential A*exp(Bx)'''
        
        sumy = np.sum(y)
        sumx2y = np.sum((x**2)*y)
        sumxy = np.sum(x*y)
        sumxy2 = sumxy**2
        lny = np.log(y)
        sumylny = np.sum(y*lny)
        sumxylny = np.sum(x*y*lny)
         
        den =  sumy * sumx2y - sumxy2
        
        a = (sumx2y*sumylny - sumxy*sumxylny)/den
        b = (sumy*sumxylny - sumxy*sumylny)/den
        
        return np.exp(a), b
    
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
        
    def get_loc_std(self,data=None,window=8):
        '''Calculate the local (rolling) standard deviation of data with the
        specified window, along axis 0.
        
        Parameters
        ----------
        data: (NT, N) or (N,) array_like (optional)
            If None, the data contained in the object is used. If data is
            passed, the local standard deviation of that array is returned.
            Default: None.
        window: int (optional)
            Size of the rolling window. Default: 8.
            
        Returns
        -------
        loc_std: (N,) array_like or scalar
            Local standard deviation. Output shape depends on data input.
        '''
        
        if len(data)==0: return 0.0
        
        # Check if the cached loc_std can be useful
        if data is None and self.loc_std_cached is not None:
            if self.loc_std_cached["window"]==window:
                return self.loc_std_cached["loc_std"]
        
        if data is None: 
            data = self.data
            data_was_none = True
        else: 
            data = np.array(data)
            data_was_none = False

        if len(data.shape)>1:        
            loc_std = np.zeros(data.shape[1])
            for j in np.arange(data.shape[1]):
                loc_std[j] = np.sqrt(np.median(np.var(self.rolling_window(data[:,j], window), axis=-1)))
        else:
            loc_std = np.sqrt(np.median(np.var(self.rolling_window(data, window), axis=-1)))
                
        
        if data_was_none:
            self.loc_std_cached = {"window":window,"loc_std":loc_std}
        
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
    
    @staticmethod
    def get_causal_sg(n,poly):
        order = np.arange(poly+1)
        out = np.zeros(n)
        for i in order:
            fil = sg.get_1D_filter(n,poly,i)
            if i%2==1: fil = fil[::-1]
            out += fil*(((n-1)//2)**i)
        return out
        
    @staticmethod
    def get_holoborodko_filter(n,poly):
        ker = np.zeros(n)
        m = (n-1)//2
        ks = np.arange(-m,m)
        for k in ks:
            ker[k+m] = (3*m-1-2*k**2)/(2*m-1)/(2**(2*m))*comb(2*m,m+k)
        return ker
        
        #if n==11 and poly==5:
        #    ker = np.array([-4,-20,-20,80,280,392,280,80,-20,-20,-4])/1024.
        #    return ker
        
    @staticmethod
    def remove_outliers(y,std_th=2.):
        y2 = y.copy()
        dy = np.diff(y)
        outls = np.where(dy>std_th*np.std(dy))[0]
        
        for outl in outls:
            if outl<len(y)-1:
                y2[outl] = 0.5*(y2[outl-1]+y2[outl+1])
            else:
                y2[outl] = y2[outl-1]
                
        return y2
    
    @classmethod
    def get_matchless_nan_th_from_file(cls,folder):
        fname = folder+cls.matchless_nan_th_fname
        
        if os.path.isfile(fname):
            f = open(fname,"r")
            l = f.readline()
            if l[-1]=="\n": l=l[:-1]
            mnt = float(l)
        else:
            mnt = None
        
        return mnt
        
        
