import numpy as np
import os
import json
import wormdatamodel as wormdm

class recording:

    # Acquisition parameters
    dt = 0.005
    piezoFrequency = 3.0
    
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
    
    # Conventions on filenames
    filenameFrames = "sCMOS_Frames_U16_1024x512.dat" #'frames_1024x512xU16.dat'
    filenameFramesDetails = "framesDetails.txt"
    filenameOtherFrameSynchronous  = "other-frameSynchronous.txt"
    filenameZDetails = "zScan.json"
    filenameOptogeneticsTwoPhoton = "pharosTriggers.txt"
    
    def __init__(self, foldername, legacy=False, rectype="3d"):
        '''
        Class constructor: Do not load all the data from the constructor, so 
        that the class can be used also in a light-weight mode.
        '''
        if foldername[-1] == "/":
            self.foldername = foldername
        else:
            self.foldername = foldername+"/"
        self.filename = self.foldername+self.filenameFrames
        
        self.legacy = legacy
        self.rectype = rectype
        
        # Load extra information
        self.load_extra()
        
        # Load optogenetics information. Autodetects if present
        self.load_optogenetics()
        
        self.frameBufferIndexes = np.array([])
        self.frame = np.array([])
    
    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        del self.frame
        
    def load(self, *args, **kwargs):
        if self.rectype=="3d":
            self._load_3d(*args, **kwargs)
        elif self.rectype=="2d":
            self._load_2d(*args, **kwargs)
    
    def free_memory(self):
        del self.frame
        self.frame = np.array([])
    
    def _load_3d(self, startFrame=0, stopFrame=-1, startVolume=0, nVolume=-1, 
                jobMaxMemory=100*2**30):
        
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
                self.load_frames_legacy(startFrame, self.frameN)
            else:
                self.load_frames(startFrame, self.frameN)
                
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
        
    def _load_2d(self, startFrame=0, stopFrame=-1, jobMaxMemory=100*2**30):
               
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
                self.load_frames_legacy(startFrame, self.frameN)
            else:
                self.load_frames(startFrame, self.frameN)
            
        
    def load_frames(self, startFrame=0, frameN=0):
        '''
        Loads the specified frames into the buffer.
        '''
        # Open file
        self.file = open(self.foldername+self.filenameFrames, 'br')
        # Go to desired start frame
        self.file.seek(self.frameSizeBytes*startFrame)
        
        self.frame = np.fromfile(self.file, dtype=self.frameDType, 
                        count=frameN*self.frameSize).reshape(
                        frameN,self.channelN,
                        self.channelSizeY,self.channelSizeX)
                        
        self.filePosition = self.file.tell()
        # Close file
        self.file.close()
        self.frameLastLoaded = startFrame+frameN
        
        self.frameBufferIndexes = np.arange(startFrame,
                                        startFrame+frameN).astype(np.uint32)
                                        
    def load_frames_legacy(self, startFrame=0, frameN=0):
        '''
        Loads the specified frames into the buffer from a file in which the
        image is stored line0R,line0G,line1R,line1G,...
        '''
        
        #self.file.close()
        self.frame = 3*np.ones(frameN*self.frameSize, dtype=np.uint16).reshape( #TODO TODO TODO TODO TODO TODO
                                        frameN,self.channelN,
                                        self.channelSizeY,self.channelSizeX)
        wormdm.data.load_frames_legacy(self.foldername+self.filenameFrames, 
                                   (np.int32)(startFrame), (np.int32)(frameN), 
                                   (np.int32)(self.channelSizeX), (np.int32)(self.channelSizeY),
                                   self.frame)
        #self.file = open(self.foldername+self.filenameFrames, 'br')
        
        self.frameLastLoaded = startFrame+frameN
        
        self.frameBufferIndexes = np.arange(startFrame,
                                        startFrame+frameN).astype(np.uint32)
                                        
    def load_extra(self):
        if self.rectype=="3d":
            self._load_extra_3d()
        elif self.rectype=="2d":
            self._load_extra_2d()
        
    def _load_extra_3d(self):
        '''
        Load extra information recorded.
        '''
        framesDetails = np.loadtxt(self.foldername+self.filenameFramesDetails,skiprows=1).T
        frameTime = framesDetails[0]
        frameCount = framesDetails[1].astype(int)

        frameSync = np.loadtxt(self.foldername+self.filenameOtherFrameSynchronous,skiprows=1).T
        frameCountDAQ = frameSync[0].astype(int)
        latencyShift = 0
        if latencyShift!=0:
            frameCountDAQ += latencyShift
            print("Applying latency shift "+str(latencyShift))
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

        frameCount = frameCountCorr
        
        # Get the volume to which each frame in FrameDetails belongs from the DAQ data
        volumeIndex = np.ones_like(frameCount)*(-10)
        self.volumeDirection = np.empty(len(frameCount),dtype=np.int8)
        self.volumeDirection[:] = np.nan
        self.Z = np.empty(len(frameCount),dtype=np.float)
        self.Z[:] = np.nan
        for i in np.arange(len(frameCount)):
            try:
                indexInDAQ = np.where(frameCountDAQ == frameCount[i])
                volumeIndex[i] = volumeIndexDAQ[indexInDAQ]
                self.Z[i] = ZDAQ[indexInDAQ]
                self.volumeDirection[i] = (volumeDirectionDAQ[indexInDAQ]+1)//2
            except:
                pass
                
        # Interpolate missing values for Z and volumeDirection
        # TODO This however, assumes that there are no gaps in frameCount
        nans, x = np.isnan(self.Z), lambda z: z.nonzero()[0]
        self.Z[nans]= np.interp(x(nans), x(~nans), self.Z[~nans])
        self.volumeDirection[nans] = np.interp(x(nans), x(~nans), self.volumeDirection[~nans]).astype(np.int8)
        
        # Use the derivative of Z to determine the volumeDirection, instead of
        # the output of the differentiator. The latter has some problems...
        # Maybe you should change the 
        if self.legacy:
            print("Using the derivative of Z to determine the volume direction and the volume index, instead of the output of the differentiator. (Legacy)")
            smn=4
            sm = np.ones(smn)/smn
            Z_sm = np.copy(self.Z)
            Z_sm = np.convolve(self.Z,sm,mode="same")
            self.volumeDirection[:-1] = np.sign(np.diff(Z_sm))
            self.volumeDirection[-1] = self.volumeDirection[-2]
            self.volumeDirection = (-self.volumeDirection+1)//2
            volumeIndex = np.cumsum(np.absolute(np.diff(self.volumeDirection)))
            
        # Get the first frame of each volume as indexes in the list of frames 
        # that have been saved, regardless of dropped frames, so that you can
        # use these directly to load the files. frameTime, the absolute 
        # frameCount, and volumeIndex (not volumeIndexDAQ) can be indexed using
        # self.volumeFirstFrame.
        self.volumeFirstFrame = np.where(np.diff(volumeIndex)==1)[0] 
        self.nVolume = len(self.volumeFirstFrame)-2
        
        # Get the details about the Z scan. The values in the DAQ file are in V.
        # The file contains also the etlCalibrationMindpt and Maxdpt, which 
        # correspond to the diopters corresponding to 0 and 5 V, respectively.
        try:
            fz = open(self.foldername+self.filenameZDetails)
            zDetails = json.load(fz)
            fz.close()
            zUmOverV = 1./zDetails["V/um"]
        except:
            zUmOverV = 1./0.05
        
        # Build the list ZZ of Z split in different volumes
        self.ZZ = []
        for k in np.arange(len(self.volumeFirstFrame)-1):
            frame0 = self.volumeFirstFrame[k]
            framef = self.volumeFirstFrame[k+1]
            zeta = self.Z[frame0:framef]*zUmOverV*self.framePixelPerUm
            # TODO is this really necessary? Isn't it already in it?
            #zeta = zeta - np.average(zeta)
            #if self.volumeDirection[frame0] == 0: zeta *= -1
            self.ZZ.append(zeta)

        #self.stackIdx = np.array(sio.loadmat(self.foldername+'hiResData.mat')['dataAll'][0][0][4])
        #self.stackIdx = self.stackIdx.reshape(self.stackIdx.shape[0])
        
    def _load_extra_2d(self):
        framesDetails = np.loadtxt(self.foldername+self.filenameFramesDetails,skiprows=1).T
        self.T = framesDetails[0]
        self.frameCount = framesDetails[1].astype(int)
        
        
    def load_optogenetics(self):
        if os.path.isfile(self.foldername+self.filenameOptogeneticsTwoPhoton):
            self.optogeneticsType = "twoPhoton"
            optogeneticsF = open(self.foldername+self.filenameOptogeneticsTwoPhoton)
            Line = optogeneticsF.readlines()
            optogeneticsF.close()
            if Line[0] == "frameCount\tnPulses\trepRateDivider\toptogTargetX\toptogTargetY\toptogTargetZ\toptogTargetXYSpace\toptogTargetZSpace\toptogTargetZDevice\tTime\n":
                Line.pop(0)
                if Line[-1]=="": Line.pop(-1)
                self.optogeneticsN = len(Line)
                self.optogeneticsFrameCount = np.zeros(self.optogeneticsN, dtype=np.int)
                self.optogeneticsNPulses = np.zeros(self.optogeneticsN, dtype=np.int)
                self.optogeneticsRepRateDivider = np.zeros(self.optogeneticsN, dtype=np.int)
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
        else:
            self.optogeneticsType = "None"
            
    def _get_memoryEstimatedUsage(self):
        '''
        You will need memory for:
        - the buffer itself, for the volumes on which
        - the parallel for will be working on
        '''
        
        return self.frameN * self.frameSizeBytes * 1.1
        
    def _get_memoryUsagePerVolume(self, nFrameInVol=None):
        if nFrameInVol is None:
            nFrameInVol = np.max(np.diff(self.volumeFirstFrame))
        return nFrameInVol*self.frameSizeBytes*1.3
        
    def get_volume(self, i):
        '''
        Returns the i-th volume, either taking it from the buffer or loading it
        from file.
        '''
        #FramesIdx, = np.where(self.stackIdx == i)
        #firstFrame = FramesIdx[0]
        #lastFrame = FramesIdx[-1]
        firstFrame = self.volumeFirstFrame[i]
        lastFrame = self.volumeFirstFrame[i+1]
        print(firstFrame,lastFrame)
        #print(firstFrame,lastFrame)
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
                
                        
    def load_frame(self, k,i):
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
