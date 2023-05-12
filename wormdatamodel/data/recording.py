import numpy as np
import os
import json
import pickle
import wormdatamodel as wormdm

class recording:
    '''Class representing a sequence of images/frames in time. The images can be
    composed of multiple channels (although the implementation currently has
    this hardcoded to 2 channels), and the sequence can be a simple video or
    a volumetric recording.
    
    At initialization, only the metadata of the recording is loaded, so that
    the class can be used in a lightweight mode.  Once the object is created, 
    the frames can be loaded in memory with the method load() passing 
    startFrame and stopFrame or, more commonly, startVolume and nVolume.
    
    Given that the frames can take up a lot of memory, the class can be used
    with contexts as in::
        with wormdatamodel.data.recording(folder) as rec:
            rec.load(startVolume=i, nVolume=N)
    so that memory is freed once the script exits the context. Alternatively,
    the memory associated with the frame buffer can be freed manually using
    the method free_memory(). 
    
    Useful methods
    --------------
    load(): to load frames, and get_events(): to return metadata about events 
    that happened during the measurement, like optogenetics stimulations. See
    methods documentation for more details
        
    Attributes
    ----------
    frame: numpy array
        Contains the loaded frames currently in memory.
    frameN: int
        Number of frames in memory.
    frameTime: numpy array
        Time of each frame in the whole recording (in the whole file, not only 
        in memory).
    frameCount: numpy array
        Absolute index (count) of the frames in the whole recording.
    Z: numpy array
        z position of each frame in the whole recording. For a volume-split 
        version of this array, see ZZ.
    nVolume: int
        Number of volumes in the whole recording .
    volumeIndex: numpy array
        Index of the volume to which each frame (in the whole recording) belongs
        to.
    volumeFirstFrame: numpy array 
        Array with the indices of the first frames of each volume.
    volumeDirection: numpy array
        Direction of each volume in the whole recording (upwards or downwards
        scan).
    ZZ: list of numpy arrays 
        Contains the z coordinate of the frames in each volume. Index as
        ZZ[volume_index][frame_index_in_volume].
    optogenetics: various
        Attributes with names starting with optogenetics contain details about
        optogenetics stimulations. They are better accessed through the method
        get_events() (see below), which returns a dictionary containing that
        information.

    '''
    
    # Frame buffer and other useful attributes
    frame = np.array([])
    frameN = 0   
     
    frameTime = np.array([])
    frameCount = np.array([])
    Z = np.array([])
    
    nVolume = 0
    volumeIndex = np.array([])
    volumeFirstFrame = np.array([])
    volumeDirection = np.array([])
    ZZ = []  
    
    # Acquisition parameters
    dt = 0.005
    piezoFrequency = 3.0
    latencyShift = 0
    rectype = None
    legacy = None
    
    channelSizeX = np.uint16
    channelSizeY = np.uint16
    channelN = np.uint16
    frameBitDepth = np.uint16
    
    channelSizeX = 512 # pixels
    channelSizeY = 512 # pixels
    channelSize = channelSizeX * channelSizeY
    channelN = 2
    frameSize = channelSize * channelN
    frameBitDepth = 16 # bits
    frameDType = np.uint16
    frameSizeBytes = frameSize * frameBitDepth // 8
    frameBinning = 1
    frameCountOffset = 100
    framePixelOffset = frameBinning*frameCountOffset
    
    frameUmPerPixel = 0.42
    framePixelPerUm = 1./frameUmPerPixel
    
    # State of memory buffer for frames
    memorySaturated = False
    frameBufferLock = False
    frameLastLoaded = -1
    frameBufferPosition = 0
    filePosition = 0
    
    referenceVolume = np.array([])
    
    # Conventions on filenames
    foldername = None
    folder = None #alias for foldername
    filename = None
    loaded_from_cache = False
    filenamePickle = "recording.pickle"
    filenameFrames = "sCMOS_Frames_U16_1024x512.dat" #'frames_1024x512xU16.dat'
    filenameFramesDetails = "framesDetails.txt"
    filenameOtherFrameSynchronous  = "other-frameSynchronous.txt"
    filenameZDetails = "zScan.json"
    filenameOptogeneticsTwoPhoton = "pharosTriggers.txt"
    
    # Optogenetics
    optogeneticsN = 0
    optogeneticsFrameCount = np.zeros(optogeneticsN, dtype=int)
    optogeneticsNPulses = np.zeros(optogeneticsN, dtype=int)
    optogeneticsRepRateDivider = np.zeros(optogeneticsN, dtype=int)
    optogeneticsNTrains = np.zeros(optogeneticsN, dtype=int)
    optogeneticsTimeBtwTrains = np.zeros(optogeneticsN)
    optogeneticsTargetX = np.zeros(optogeneticsN)
    optogeneticsTargetY = np.zeros(optogeneticsN)
    optogeneticsTargetZ = np.zeros(optogeneticsN)
    optogeneticsTargetXYSpace = ["None"]*optogeneticsN
    optogeneticsTargetZSpace = ["None"]*optogeneticsN
    optogeneticsTargetZDevice = ["None"]*optogeneticsN
    optogeneticsTime = ["None"]*optogeneticsN
    
    def __new__(cls, *args, **kwargs):
        
        if len(args)>0 or "foldername" in kwargs.keys():
            try: foldername = args[0]
            except: foldername = kwargs["foldername"]
            if foldername[-1] == "/":
                foldername = foldername
            else:
                foldername = foldername+"/"

            if os.path.isfile(foldername+cls.filenamePickle):
                f = open(foldername+cls.filenamePickle,"rb")
                inst = pickle.load(f)
                f.close()
                inst.loaded_from_cache = True
            else:
                inst = super(recording, cls).__new__(cls)
        else:
            inst = super(recording, cls).__new__(cls)
            
        return inst
            
    
    def __init__(self, foldername=None, legacy=False, rectype=None, settings={}):
        '''The class constructor does not load all the data, so that the class 
        can be used also in a light-weight mode. If the recording type (2d or 
        3d) is not specified, the constructor determines it based on the file
        structure, then loads the metadata of the recording (including the data
        relative to optogenetics stimulations, if present).
        
        Parameters
        ----------
        foldername: string
            Folder containing the recording. The file names must follow the
            convention detailed by the default values of the filename
            attributes.
        legacy: bool (optional)
            Set to True if the recording was taken on the old whole brain 
            imager. This is needed, e.g., to determine how the frames are stored
            in the file, with each frame in each channel being contiguous
            (legacy=False) or with line0R,line0G,line1R,line1G... 
            Default: False.
        rectype: string (optional)
            Can be "3d" or "2d". If it is not set, the constructor tries to
            determine it based on the file structure. Default: None.
        settings: dict (optional)
            Contains additional settings. Currently only the key "latencyShift"
            is used, which specifies if there is a delay between the z position
            of the device used (piezo motor or tunable lens) and its monitor
            output. It depends, e.g., on whether the piezo and the objective
            are mounted vertically or horizontally.
        '''
        if foldername is not None and not self.loaded_from_cache:
            if foldername[-1] == "/":
                self.foldername = foldername
            else:
                self.foldername = foldername+"/"
            self.folder = self.foldername
            self.filename = self.foldername+self.filenameFrames
            
            self.legacy = legacy
            
            if rectype == None:
                if os.path.isfile(foldername+self.filenameZDetails): 
                    self.rectype = "3d"
                else:
                    self.rectype = "2d"
                print("Assuming it is a "+self.rectype+" recording.")
            else:
                self.rectype = rectype
                
                
            try:
                self.latencyShift = settings["latencyShift"]
            except:
                self.latencyShift = 0
                
            try:
                self.manualZUmOverV = settings["zUmOverV"]
            except:
                self.manualZUmOverV = None
            
            # Load extra information
            self.load_extra()
            
            # Load optogenetics information. Autodetects if present
            self.load_optogenetics()
            
            self.frameBufferIndexes = np.array([])
            self.frame = np.array([])
            
            # Cache the object for faster loading as pickle
            pickle_file = open(self.foldername+self.filenamePickle,"wb")
            pickle.dump(self,pickle_file)
            pickle_file.close()
    
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        del self.frame
        
    def load(self, *args, **kwargs):
        if self.rectype=="3d":
            im = self._load_3d(*args, **kwargs)
        elif self.rectype=="2d":
            im = self._load_2d(*args, **kwargs)
        if "standalone" in kwargs.keys():
            if kwargs["standalone"]:
                return im
    
    def free_memory(self):
        del self.frame
        self.frame = np.array([])
    
    def _load_3d(self, startFrame=0, stopFrame=-1, startVolume=0, nVolume=-1, 
                jobMaxMemory=100*2**30, standalone=False):
        '''Load the frames from a 3D recording. Specify either start and stop
        frames or start and number of volumes. Will not load anything if the
        estimated memory usage exceeds the maximum job memory limit. 
        All parameters are optional. If nothing is passed, the
        function attempts to load the whole file.
        
        The frames are directly loaded in self.frame and not returned.
        
        Parameters
        ----------
        startFrame: int (optional)
            First frame to load, in the file reference frame (not absolute
            frame index). Used only if nVolume is not set. Default: 0.
        stopFrame: int (optional)
            Last frame to load, in the file reference frame (not absolute 
            frame index). Used only if nVolume is not set. 
            Default: -1, i.e. until end of file.
        startVolume: int (optional)
            First volume to load. The 0th volume is the first complete volume
            in the recording. Used only if nVolume is set. Default: 0.
        nVolume: int (optional)
            Number of volumes to load. Default: -1, corresponding to not set.
        jobMaxMemory: scalar (optional)
            Sets the limit of memory that can be used. Default: 100 GB.
        '''
        
        if nVolume!=-1:
            startFrame = self.volumeFirstFrame[startVolume]
            stopFrame = self.volumeFirstFrame[startVolume+nVolume]
        
        if stopFrame==-1: 
            # Calculate number of frames contained in the file and subtract 
            # initial frames skipped
            self.frameN = os.stat(self.filename).st_size / self.frameSizeBytes
            self.frameN -= startFrame
            
        else:
            self.frameN = stopFrame - startFrame
        
        # Get an estimate of the memory that will be used in the analysis
        self.memoryEstimatedUsage = self._get_memoryEstimatedUsage()

        # Now you should check that you are not loading and allocating more data
        # that can be contained in RAM. If you think you're hitting the limit
        # load up to a maximum number of frames and store the position you reach
        # in the file.
        
        #self.frameBufferSize = int(round(min(self.memoryEstimatedUsage, jobMaxMemory))) // self.frameSizeBytes
        if self.memoryEstimatedUsage > jobMaxMemory:
            self.memorySaturated = True
            print("You tried to load more data than memory available. Nothing done.")
        else:
            # Initialize look-up table for actual frame indexes inside the buffer
            self.frameBufferIndexes = np.zeros(self.frameN, 
                                                dtype=np.uint32)
            
            # Load the frames
            if self.legacy==True:
                im = self.load_frames_legacy(startFrame, self.frameN, standalone)
            else:
                im = self.load_frames(startFrame, self.frameN, standalone)
        
            if standalone:
                return im
                
        # The following is for future upgrades
        # self.frames will act as a circular buffer, so you need a system to 
        # lock the frames on which the program is working, so that they are not
        # overwritten.
        # Initialize empty numpy array for the frame buffer [ch, i, x, y]
        #self.frame = np.empty((self.frameBufferSize, self. channelN,
        #                        self.channelSizeX, self.channelSizeY), 
        #                        dtype=np.uint16)
        
        # Initialize lock of the buffer, to avoid loading more data when you
        # cannot. The simple version is to get a single lock on everything.
        #self.frameBufferLocks = np.zeros(self.frameBufferSize, dtype=np.bool)
        
        
    def _load_2d(self, startFrame=0, stopFrame=-1, startVolume=0, nVolume=-1, jobMaxMemory=100*2**30, standalone=False):
        '''Load the frames from a 2D recording. Specify either start and stop
        frames or start and number of volumes. In this case, the volume notation
        is provided to keep a uniform notation across 2D and 3D recordings.
        Each frame corresponds to one volume. Will not load anything if the
        estimated memory usage exceeds the maximum job memory limit. 
        All parameters are optional. If nothing is passed, the
        function attempts to load the whole file.
        
        The frames are directly loaded in self.frame and not returned.
        
        Parameters
        ----------
        startFrame: int (optional)
            First frame to load, in the file reference frame (not absolute
            frame index). Used only if nVolume is not set. Default: 0.
        stopFrame: int (optional)
            Last frame to load, in the file reference frame (not absolute 
            frame index). Used only if nVolume is not set. 
            Default: -1, i.e. until end of file.
        startVolume: int (optional)
            First volume to load. The 0th volume is the first complete volume
            in the recording. Used only if nVolume is set. Default: 0.
        nVolume: int (optional)
            Number of volumes to load. Default: -1, corresponding to not set.
        jobMaxMemory: scalar (optional)
            Sets the limit of memory that can be used. Default: 100 GB.
        '''
        
        
        if nVolume != -1:
            startFrame = startVolume
            stopFrame = startFrame + nVolume
             
        if stopFrame==-1: 
            # Calculate number of frames contained in the file and subtract 
            # initial frames skipped
            self.frameN = os.stat(self.filename).st_size / self.frameSizeBytes
            self.frameN -= startFrame
            self.frameN = (int)(self.frameN)
            
        else:
            self.frameN = stopFrame - startFrame
        
        # Get an estimate of the memory that will be used in the analysis
        self.memoryEstimatedUsage = self._get_memoryEstimatedUsage()

        # Now you should check that you are not loading and allocating more data
        # that can be contained in RAM. If you think you're hitting the limit
        # load up to a maximum number of frames and store the position you reach
        # in the file.
        
        #self.frameBufferSize = int(round(min(self.memoryEstimatedUsage, jobMaxMemory))) // self.frameSizeBytes
        if self.memoryEstimatedUsage > jobMaxMemory:
            self.memorySaturated = True
            print("You tried to load more data than memory available. Nothing done.")
        else:
            # Initialize look-up table for actual frame indexes inside the buffer
            self.frameBufferIndexes = np.zeros(self.frameN, 
                                                dtype=np.uint32)
            
            # Load the frames
            if self.legacy==True:
                self.load_frames_legacy(startFrame, self.frameN, standalone)
            else:
                self.load_frames(startFrame, self.frameN, standalone)
                
            if standalone:
                return im
            
        
    def load_frames(self, startFrame=0, frameN=0, standalone=False):
        '''Loads the specified frames into the class buffer, from a file in which
        each frame in each channel is contiguously stored (line0R, line1R, ...,
        line0G, line1G, ...)
        
        Parameters
        ----------
        startFrame: int (optional)
            First frame to load, in the file reference frame (not absolute frame
            count). Default: 0.
        frameN: int (optional)
            Number of frames to load.
        '''
        
        # Open file
        self.file = open(self.foldername+self.filenameFrames, 'br')
        # Go to desired start frame
        self.file.seek(self.frameSizeBytes*startFrame)
        
        im = np.fromfile(self.file, dtype=self.frameDType, 
                        count=frameN*self.frameSize).reshape(
                        frameN,self.channelN,
                        self.channelSizeY,self.channelSizeX)
                        
        self.filePosition = self.file.tell()
        # Close file
        self.file.close()
        
        if not standalone:
            self.frame = im
            self.frameLastLoaded = startFrame+frameN
            
            self.frameBufferIndexes = np.arange(startFrame,
                                            startFrame+frameN).astype(np.uint32)
            
            return None
        else:
            return im
                                        
    def load_frames_legacy(self, startFrame=0, frameN=0, standalone=False):
        '''Loads the specified frames into the buffer from a file in which the
        image is stored line0R,line0G,line1R,line1G,...
        
        Parameters
        ----------
        startFrame: int (optional)
            First frame to load, in the file reference frame (not absolute frame
            count). Default: 0.
        frameN: int (optional)
            Number of frames to load.
        '''
        
        # Pre-allocate the memory for the frames          
        im = np.empty((frameN,self.channelN,
                               self.channelSizeY,self.channelSizeX),
                               dtype=np.uint16)
        # Load                       
        wormdm.data.load_frames_legacy(self.foldername+self.filenameFrames, 
                                   (np.int32)(startFrame), (np.int32)(frameN), 
                                   (np.int32)(self.channelSizeX), 
                                   (np.int32)(self.channelSizeY),
                                   im)
        
        if not standalone:
            self.frame = im
            self.frameLastLoaded = startFrame+frameN
            
            self.frameBufferIndexes = np.arange(startFrame,
                                            startFrame+frameN).astype(np.uint32)
                                            
            return None
        else:
            return im
                                        
    def load_extra(self):
        '''Load metadata for the recording, implicitly choosing the type of
        metadata depending on the recording type (2D or 3D). Loads directly in
        the class variables, does not return anything.
        '''
        if self.rectype=="3d":
            self._load_extra_3d()
        elif self.rectype=="2d":
            self._load_extra_2d()
        
    def _load_extra_3d(self):
        '''Load metadata for 3D recordings. The function builds a correspondence
        between frame index in the file reference frame and the absolute frame
        count, assigns each frame to a volume, and to a z position.
        '''
        # Load the details of the frames that are downloaded from the camera
        # together with the frames: for each frame in the sequence of saved
        # frames there is one entry in this file containing its absolute count
        # from the beginning of the acquisition (not the saving) and its 
        # timestamp.
        framesDetails = np.loadtxt(self.foldername+self.filenameFramesDetails,skiprows=1).T
        self.frameTime = framesDetails[0]
        a, b = np.unique(np.diff(self.frameTime), return_counts=True)
        self.dt = round(a[np.argmax(b)],3)
        frameCount = framesDetails[1].astype(int)
        
        # Load the details of the frames that are acquired synchrounously to
        # the camera triggers. Each entry contains the value of the counter 
        # counting the triggers (framesCountDAQ), corresponding to the absolute
        # count downloaded from the camera.
        frameSync = np.loadtxt(self.foldername+self.filenameOtherFrameSynchronous,skiprows=1).T
        frameCountDAQ = frameSync[0].astype(int)
        #latencyShift = 0
        if self.latencyShift!=0:
            frameCountDAQ += self.latencyShift
            print("Applying latency shift: "+str(self.latencyShift)+" frames.")
        volumeIndexDAQ = frameSync[3].astype(int)
        volumeDirectionDAQ = frameSync[2].astype(int)
        ZDAQ = frameSync[1]
        
        # Correct frameCount: sometimes the frameCount jumps by 2 at one step
        # and by 0 at the next one. TODO: check whether the index in the buffer
        # changes accordingly, or if I'm saving the same frame twice.
        frameCountCorr = np.copy(frameCount)
        dframeCount = np.diff(frameCount)
        
        for i in np.arange(len(frameCount)-1):
            if(dframeCount[i]==0 and dframeCount[i-1]==2):
                frameCountCorr[i] -= 1

        self.frameCount = frameCountCorr
        
        # Get the volume to which each frame in FrameDetails belongs from the DAQ data
        volumeIndex = np.ones_like(frameCount)*(-10)
        self.volumeDirection = np.empty(len(frameCount),dtype=float)#FIXME,dtype=np.int8)
        self.volumeDirection[:] = np.nan
        self.Z = np.empty(len(frameCount),dtype=float)
        self.Z[:] = np.nan
        # Debugging arrays
        #self.fcountfound = np.zeros(len(frameCount))
        #self.DAQneighbor_adjacent = np.zeros(len(frameCount))
        for i in np.arange(len(frameCount)):
            # Find the index of the entry in the camera-trigger-synchronous
            # data that corresponds to the absolute count frameCount[i].
            indexInDAQ = np.where(frameCountDAQ == frameCount[i])
            #self.fcountfound[i] = indexInDAQ[0].shape[0]
            
            # Extract the information from that entry, only if the DAQ counter
            # saw that frame count only once, and if the neighboring counts
            # are adjacent (+-1).
            # If there is no matching saved frame (dropped), pass and leave
            # nan entries. They will be interpolated below.
            
            if indexInDAQ[0].shape[0] == 1:
                if indexInDAQ[0][0]+1 < len(frameCountDAQ):
                    neighbor_adjacent = ((frameCountDAQ[indexInDAQ[0][0]]-frameCountDAQ[indexInDAQ[0][0]-1])==1) and ((frameCountDAQ[indexInDAQ[0][0]+1]-frameCountDAQ[indexInDAQ[0][0]])==1)
                    #self.DAQneighbor_adjacent[i] =  neighbor_adjacent
                    
                    if neighbor_adjacent:
                        volumeIndex[i] = volumeIndexDAQ[indexInDAQ]
                        self.Z[i] = ZDAQ[indexInDAQ]
                        self.volumeDirection[i] = (volumeDirectionDAQ[indexInDAQ]+1)//2
                
        # Interpolate missing values for Z and volumeDirection
        # If there are gaps in frameCount (dropped frames) inside the volumes,  
        # the interpolation of the Z will produce steeper regions, that remain,
        # however, locally monotonic. If the gaps are at the volume boundaries,
        # this will move the change of sign of the derivative into one of the
        # neighboring volumes.
        nans, x = np.isnan(self.Z), lambda z: z.nonzero()[0]
        self.Z[nans]= np.interp(x(nans), x(~nans), self.Z[~nans])
        self.volumeDirection[nans] = np.interp(x(nans), x(~nans), self.volumeDirection[~nans]).astype(float)#FIXME astype(np.int8)
        volumeIndex[nans]= np.interp(x(nans), x(~nans), volumeIndex[~nans])
        
        # Use the derivative of Z to determine the volumeDirection, instead of
        # the output of the differentiator. The latter has some problems...
        # Maybe you should change the 
        if self.legacy:
            print("Using the derivative of Z to determine the volume direction and the volume index, instead of the output of the differentiator. (Legacy)")
            smn=4
            sm = np.ones(smn)/smn
            Z_sm = np.copy(self.Z)
            #FIXME Z_sm = np.convolve(self.Z,sm,mode="same")
            self.volumeDirection[:-1] = np.sign(np.diff(Z_sm))
            self.volumeDirection[-1] = self.volumeDirection[-2]
            self.volumeDirection = (-self.volumeDirection+1)//2
            volumeIndex[1:] = np.cumsum(np.absolute(np.diff(self.volumeDirection)))
            volumeIndex[0] = volumeIndex[1]

        # Subtract 1 because volume 0 has to be the first complete volume.
        self.volumeIndex = (volumeIndex-volumeIndex[0]).astype(int)-1
            
        # Get the first frame of each volume as indexes in the list of frames 
        # that have been saved, regardless of dropped frames, so that you can
        # use these directly to load the files. frameTime, the absolute 
        # frameCount, and volumeIndex (not volumeIndexDAQ) can be indexed using
        # self.volumeFirstFrame.
        self.volumeFirstFrame = np.where(np.diff(volumeIndex)==1)[0]#+1
        self.nVolume = len(self.volumeFirstFrame)-2
        a, b = np.unique(np.diff(self.volumeFirstFrame), return_counts=True)
        self.Dt = round(a[np.argmax(b)],3)*self.dt
        
        # Get the details about the Z scan. The values in the DAQ file are in V.
        # The file contains also the etlCalibrationMindpt and Maxdpt, which 
        # correspond to the diopters corresponding to 0 and 5 V, respectively.
        #try:
        if os.path.isfile(self.foldername+self.filenameZDetails):
            fz = open(self.foldername+self.filenameZDetails)
            zDetails = json.load(fz)
            fz.close()
            self.zUmOverV = 1./zDetails["V/um"]
            if "etlCalibrationMindpt" in zDetails.keys():
                self.etlCalibrationMindpt = zDetails["etlCalibrationMindpt"]
                self.etlCalibrationMaxdpt = zDetails["etlCalibrationMaxdpt"]
                self.etlVMin = zDetails["etlVMin"]
                self.etlVMax = zDetails["etlVMax"]
                self.etlDptOverUm = zDetails["etl dpt/um"]
                self.etlVOverDpt = (self.etlVMax-self.etlVMin) / (self.etlCalibrationMaxdpt-self.etlCalibrationMindpt)
        elif self.manualZUmOverV is not None:
            self.zUmOverV = self.manualZUmOverV
            zDetails = {}
        else:
            raise ValueError("z-scan details absent. You need to pass zUmOverV in the settings of the recording object")
        #except:
        #    self.zUmOverV = 1./0.0625
            
        if "latencyShiftPermutation" in zDetails.keys():
            self.latencyShiftPermutation = zDetails["latencyShiftPermutation"]
        
        # Build the list ZZ of Z split in different volumes
        self.ZZ = []
        for k in np.arange(len(self.volumeFirstFrame)-1):
            frame0 = self.volumeFirstFrame[k]
            framef = self.volumeFirstFrame[k+1]
            zeta = self.Z[frame0:framef]*self.zUmOverV*self.framePixelPerUm
            # TODO is this really necessary? Isn't it already in it?
            #zeta = zeta - np.average(zeta)
            #if self.volumeDirection[frame0] == 0: zeta *= -1
            self.ZZ.append(zeta)

        #self.stackIdx = np.array(sio.loadmat(self.foldername+'hiResData.mat')['dataAll'][0][0][4])
        #self.stackIdx = self.stackIdx.reshape(self.stackIdx.shape[0])
        
    def _load_extra_2d(self):
        '''Load metadata for a 2D recording. The function loads an absolute time
        axis, the absolute frame counts, and counts the number of volumes,
        with each volume being one single frame.
        '''
        
        framesDetails = np.loadtxt(self.foldername+self.filenameFramesDetails,skiprows=1).T
        self.T = framesDetails[0]
        a, b = np.unique(np.diff(self.T), return_counts=True)
        self.dt = round(a[np.argmax(b)],3)
        self.Dt = self.dt
        self.frameCount = framesDetails[1].astype(int)
        self.nVolume = self.frameCount.shape[0]
        self.zUmOverV = 1.0
        
        
    def load_optogenetics(self):
        '''Load optogenetics metadata. Automatically detects a couple of 
        saving configurations, that have changed in time.
        '''
        if os.path.isfile(self.foldername+self.filenameOptogeneticsTwoPhoton):
            self.optogeneticsType = "twoPhoton"
            optogeneticsF = open(self.foldername+self.filenameOptogeneticsTwoPhoton)
            Line = optogeneticsF.readlines()
            optogeneticsF.close()
            if Line[0] == "frameCount\tnPulses\trepRateDivider\toptogTargetX\toptogTargetY\toptogTargetZ\toptogTargetXYSpace\toptogTargetZSpace\toptogTargetZDevice\tTime\n":
                Line.pop(0)
                if Line[-1]=="": Line.pop(-1)
                self.optogeneticsN = len(Line)
                self.optogeneticsFrameCount = np.zeros(self.optogeneticsN, dtype=int)
                self.optogeneticsNPulses = np.zeros(self.optogeneticsN, dtype=int)
                self.optogeneticsRepRateDivider = np.zeros(self.optogeneticsN, dtype=int)
                self.optogeneticsTargetX = np.zeros(self.optogeneticsN)
                self.optogeneticsTargetY = np.zeros(self.optogeneticsN)
                self.optogeneticsTargetZ = np.zeros(self.optogeneticsN)
                self.optogeneticsTargetXYSpace = ["None"]*self.optogeneticsN
                self.optogeneticsTargetZSpace = ["None"]*self.optogeneticsN
                self.optogeneticsTargetZDevice = ["None"]*self.optogeneticsN
                self.optogeneticsTime = ["None"]*self.optogeneticsN
                
                #These won't be populated in this case
                self.optogeneticsNTrains = np.zeros(self.optogeneticsN, dtype=int)
                self.optogeneticsTimeBtwTrains = np.zeros(self.optogeneticsN)
                
                for i in np.arange(self.optogeneticsN):
                    line = Line[i]
                    sline = line.split("\t")
                    self.optogeneticsFrameCount[i] = int(sline[0])
                    self.optogeneticsNPulses[i] = int(sline[1])
                    self.optogeneticsRepRateDivider[i] = int(sline[2])
                    self.optogeneticsTargetX[i] = float(sline[3])
                    self.optogeneticsTargetY[i] = float(sline[4])
                    self.optogeneticsTargetZ[i] = float(sline[5])
                    self.optogeneticsTargetXYSpace[i] = sline[6]
                    self.optogeneticsTargetZSpace[i] = sline[7]
                    self.optogeneticsTargetZDevice[i] = sline[8]
                    self.optogeneticsTime[i] = sline[9]
                    
            elif Line[0] == "frameCount\tnPulses\trepRateDivider\tnTrains\ttimeBtwTrains\toptogTargetX\toptogTargetY\toptogTargetZ\toptogTargetXYSpace\toptogTargetZSpace\toptogTargetZDevice\tTime\n":
                Line.pop(0)
                if Line[-1]=="": Line.pop(-1)
                self.optogeneticsN = len(Line)
                self.optogeneticsFrameCount = np.zeros(self.optogeneticsN, dtype=int)
                self.optogeneticsNPulses = np.zeros(self.optogeneticsN, dtype=int)
                self.optogeneticsRepRateDivider = np.zeros(self.optogeneticsN, dtype=int)
                self.optogeneticsNTrains = np.zeros(self.optogeneticsN, dtype=int)
                self.optogeneticsTimeBtwTrains = np.zeros(self.optogeneticsN)
                self.optogeneticsTargetX = np.zeros(self.optogeneticsN)
                self.optogeneticsTargetY = np.zeros(self.optogeneticsN)
                self.optogeneticsTargetZ = np.zeros(self.optogeneticsN)
                self.optogeneticsTargetXYSpace = ["None"]*self.optogeneticsN
                self.optogeneticsTargetZSpace = ["None"]*self.optogeneticsN
                self.optogeneticsTargetZDevice = ["None"]*self.optogeneticsN
                self.optogeneticsTime = ["None"]*self.optogeneticsN
                
                for i in np.arange(self.optogeneticsN):
                    line = Line[i]
                    sline = line.split("\t")
                    if int(sline[0]) <= self.frameCount[-1]:
                        self.optogeneticsFrameCount[i] = int(sline[0])
                        self.optogeneticsNPulses[i] = int(sline[1])
                        self.optogeneticsRepRateDivider[i] = int(sline[2])
                        self.optogeneticsNTrains[i] = int(sline[3])
                        self.optogeneticsTimeBtwTrains[i] = float(sline[4])
                        self.optogeneticsTargetX[i] = float(sline[5])
                        self.optogeneticsTargetY[i] = float(sline[6])
                        self.optogeneticsTargetZ[i] = float(sline[7])
                        self.optogeneticsTargetXYSpace[i] = sline[8]
                        self.optogeneticsTargetZSpace[i] = sline[9]
                        self.optogeneticsTargetZDevice[i] = sline[10]
                        self.optogeneticsTime[i] = sline[11]
                        
                        
                        if self.optogeneticsTargetZSpace[i] == "etl0 space" and self.optogeneticsTargetZDevice[i] == "tunable lens":
                            # If the device is the etl0 tunable lens, convert z to the corresponding voltage output by the DAQ card
                            self.optogeneticsTargetZ[i] += (self.etlCalibrationMaxdpt-self.etlCalibrationMindpt)/2.0
                            self.optogeneticsTargetZ[i] *=  self.framePixelPerUm/self.etlDptOverUm
            
            # Load the "Pharos on" column from other_frameSynchronous to extract the exact timings of the triggers
            # Skip this step for 2D measurements, for which this file does not
            # exist
            if os.path.isfile(self.foldername+self.filenameOtherFrameSynchronous):
                frame_sync = np.loadtxt(self.foldername+self.filenameOtherFrameSynchronous,skiprows=1).T
                pharos_on = frame_sync[6]
                frame_count_sync = frame_sync[0]
                trigger_frame_count = frame_count_sync[np.where(np.diff(pharos_on)==1)[0]] + 1
                
                for i in np.arange(self.optogeneticsN):
                    self.optogeneticsFrameCount[i] = trigger_frame_count[np.argmin(np.absolute(trigger_frame_count-self.optogeneticsFrameCount[i]))]
        else:
            self.optogeneticsType = "None"
            
    def _get_memoryEstimatedUsage(self):
        '''Returns the estimated memory usage for the class, based on:
        - the buffer itself, for the volumes on which
        - the parallel for will be working on
        '''
        
        return self.frameN * self.frameSizeBytes * 1.1
        
    def _get_memoryUsagePerVolume(self, nFrameInVol=None):
        if nFrameInVol is None:
            nFrameInVol = np.max(np.diff(self.volumeFirstFrame))
        return nFrameInVol*self.frameSizeBytes*1.3
        
    def get_volume(self, i):
        '''Returns the i-th volume, either taking it from the buffer or loading it
        from file.
        '''
        #FramesIdx, = np.where(self.stackIdx == i)
        #firstFrame = FramesIdx[0]
        #lastFrame = FramesIdx[-1]
        firstFrame = self.volumeFirstFrame[i]
        lastFrame = self.volumeFirstFrame[i+1]
        volumeZ = np.zeros(lastFrame-firstFrame)
        
        if firstFrame in self.frameBufferIndexes:
            firstFrame = np.where(self.frameBufferIndexes==firstFrame)[0][0]
            lastFrame = np.where(self.frameBufferIndexes==lastFrame)[0][0]  
            vol = wormdm.data.volume(self.frame[firstFrame:lastFrame], 
                         volumeZ) #[i]
        else:
            vol = wormdm.data.volume(self.load_framesStandAlone(firstFrame,lastFrame-1),
                         volumeZ) #[i]
                         
        return vol
        
    def get_events(self, shift=0, shift_vol=0):
        '''Returns metadata about events happened during the measurement, like
        optogenetics stimulations. 
        
        Parameters
        ----------
        shift: int (optional)
            Shift in frames from the event.
            
        shift_vol: int (optional)
            Shift in volumes from the event.
        
        Returns
        -------
        events: dict
            Dictionary containing the metadata. For example,
            events['optogenetics'] is a dictionary with keys 'index', 'strides',
            'properties'. Index is the frame index at which the event happened,
            strides are the number of frames between two events (meant to be 
            used with irregular arrays, for example). Properties is a dictionary
            with keys 'type', 'n_pulses', 'rep_rate_divider', 'n_trains', 
            'time_btw_trains', 'target', 'target_xy_space', 'target_z_space',
            'target_z_device', 'time'.
        
        '''
        events = {}
        
        # OPTOGENETICS
        I = self.optogeneticsFrameCount.shape[0]
        
        # Find the closest frame to the trigger event, in case the exact one
        # has been dropped.
        index = np.zeros(I,dtype=np.int32)
        for i in np.arange(I):
            d = np.absolute(self.frameCount-(self.optogeneticsFrameCount[i]+shift))
            index[i] = np.argmin(d)
            
        if self.rectype == "3d":
            index = self.volumeIndex[index]
            index -= shift_vol
            
        ampl_indices = np.zeros(len(index)+2)
        ampl_indices[1:-1] = index
        ampl_indices[-1] = self.nVolume
        strides = np.diff(ampl_indices).astype(int)
            
        properties = {
            'type': 'twophoton',
            'n_pulses': self.optogeneticsNPulses,
            'rep_rate_divider': self.optogeneticsRepRateDivider,
            'n_trains': self.optogeneticsNTrains,
            'time_btw_trains': self.optogeneticsTimeBtwTrains,
            'target': np.array([
                self.optogeneticsTargetX,
                self.optogeneticsTargetY,
                self.optogeneticsTargetZ
                ]).T,
            'target_xy_space': self.optogeneticsTargetXYSpace,
            'target_z_space': self.optogeneticsTargetZSpace,
            'target_z_device': self.optogeneticsTargetZDevice,
            'time': self.optogeneticsTime
            }
        
         
        events['optogenetics'] = {'index': index, 
                                  'strides': strides,
                                  'properties': properties}
        
        return events
        
    def get_vol(self,i):
        vol = self.load(startVolume=i, nVolume=1, standalone=True)
            
        return vol
        
    def red_to_green(self,zyx):
        return wormdm.data.redToGreen(zyx,folder=self.folder)
    
    
    ## SOME FUNCTIONS FOR FUTURE IMPLEMENTATION
    
    def load_frames_onebyone(self, startFrame=0, stopFrame=-1):
        '''
        Loads the specified frames into the buffer.
        '''
        
        # Go to desired start frame
        self.file.seek(self.frameSizeBytes*startFrame)
        
        i = 0
        while self.load_frame( startFrame + i,
                (self.frameBufferPosition + i) % self.frameBufferSize ):
            i += 1
            if stopFrame != -1 and i == (stopFrame-startFrame): break

        # Update the variables storing the positions in the file and in the 
        # buffer. TODO controlla questi i+1
        self.frameBufferPosition = (self.frameBufferPosition + i +1) % \
                                   self.frameBufferSize
        
        self.filePosition = self.file.tell()
        self.frameLastLoaded = startFrame + i
                
                        
    def load_frame(self, k, i):
        '''
        Loads next frame from current position in the opened file and stores it
        at the i-th index in the self.frames buffer. Returns True if it
        successfully read a full chunk from the file, False if it did not.
        '''
        
        chunk = self.file.read(self.frameSizeBytes)
        
        if len(chunk) == self.frameSizeBytes:
        
            # Unpack bytes from file
            im = np.array(struct.unpack(str(self.frameSize)+'H', chunk))
            
            # Reshape it so that it looks like an image
            im = im.reshape((self.channelSizeX, 
                             self.channelSizeY*self.channelN))
            
            # Split the image into the channels
            for j in np.arange(self.channelN):
                self.frame[j,i,...] = \
                        im[:,j*self.channelSizeX:(j+1)*self.channelSizeX]
            
            self.frameBufferIndexes[i] = k
            
            return True
        else:
            return False
            
    def load_frameStandAlone(self, k):
        '''
        Returns a frame as a numpy array, without storing it in an attribute of
        the class itself.
        '''
        
        # Save old position in file
        position = self.filePosition
        
        # Go to this frame's position
        f.seek(k*self.frameSizeBytes)
        chunk = self.f.read(self.frameSizeBytes)

        if len(chunk) == self.frameSizeBytes:
        
            # Unpack bytes from file
            im = np.array(struct.unpack(str(self.frameSize)+'H', chunk))
            
            # Reshape it so that it looks like an image
            im = im.reshape((self.channelSizeX, 
                             self.channelSizeY*self.channelN))
            
            frame = np.empty((self. channelN, self.channelSizeX, 
                                self.channelSizeY), dtype=np.uint16)
            # Split the image into the channels
            for j in np.arange(self.channelN):
                frame[j] = im[:,j*self.channelSizeX:(j+1)*self.channelSizeX]
            
            return Frame
        else:
            return False
        
        # Go back to the previous position
        f.seek(position)
        
    def load_framesStandAlone(self, startFrame, stopFrame):
        '''
        Returns a stack of consecutive frames in the recording as a numpy array,
        without storing them in an attribute of the class itself.
        '''
        # Save old position in file
        position = self.filePosition
        
        # Initialize array for the stack
        stackLength = stopFrame - startFrame
        stack = np.zeros((self.channelN, stackLength, self.channelSizeX, 
                          self.channelSizeY), dtype=np.uint16)
        
        # Go to this frame's position
        f.seek(startFrame*self.frameSizeBytes)
        
        for i in np.arange(stackLength): 
        
            chunk = self.f.read(self.frameSizeBytes)

            if len(chunk) == self.frameSizeBytes:
            
                # Unpack bytes from file
                im = np.array(struct.unpack(str(self.frameSize)+'H', chunk))
                
                # Reshape it so that it looks like an image
                im = im.reshape((self.channelSizeX, 
                                 self.channelSizeY*self.channelN))
                
                # Split the image into the channels
                for j in np.arange(self.channelN):
                    stack[j,i,...] = \
                            im[:,j*self.channelSizeX:(j+1)*self.channelSizeX]
        
        # Go back to the previous position
        f.seek(position)
