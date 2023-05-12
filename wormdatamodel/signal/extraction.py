'''
Functions related to the extraction of signal from the frames, coded as numpy
slicing and eventual weighting.

Imports
-------
numpy
'''

import numpy as np

def _generate_box_indices(Centers, box_size=(1,3,3), Box=np.array([]), 
                          shape=[None,None,None]):
    '''
    Generate indices for slicing of an array to extract the pixels in 
    boxes/windows of given size centered on Centers.
    
    Parameters
    ----------
    Centers: numpy array of integers
        Centers[center_index, coordinate]. Coordinates are in slicing order, not
        plotting order (i.e. z,y,x for volumetric images).
    box_size: list of odd integers (optional)
        Size of the boxes centered on each Center[i]. Again, coordinates are in
        slicing order. This parameter is used if the parameter Box is not passed
        or its shape[0] is 0. Default: (1,3,3)
    Box: numpy array (optional)
        Custom box. Default: np.array([])
    shape: list of integers
        Shape of the frames array.
        
    Returns
    -------
    Indices: numpy array of integers
        Indices over the frames representing the selected box repeated over the
        Centers. Use this array to slice the frames array to extract the pixels
        in the boxes around each center.        
    '''
    
    nCenters = Centers.shape[0]
    nCoord = Centers.shape[1]
    
    if Box.shape[0]==0:
        box_size = np.array(box_size)
        nElements = np.prod(box_size)
        if np.all(box_size==np.array([5,5,5])):
            Box = np.array([
                   [-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,-2,\
                   -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,\
                   2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],
                   [-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,\
                   -2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,\
                   -2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2],
                   [-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,\
                   -2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,\
                   -2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2]
                         ])
        elif np.all(box_size==np.array([3,5,5])):
            Box = np.array([
                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                    [-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2],
                    [-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2]
                          ])
        elif np.all(box_size==np.array([3,3,3])):
            Box = np.array([
                    [-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
                    [-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1],
                    [-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1]
                    ])
        elif np.all(box_size==np.array([1,5,5])):
            Box = np.array([
                    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                    [-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,0,0,0,0,0,1,1,1,1,1,2,2,2,2,2],
                    [-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2,-2,-1,0,1,2]
                  ])
        elif np.all(box_size==np.array([1,3,3])):
            Box = np.array([
                    [0,0,0,0,0,0,0,0,0],
                    [-1,-1,-1,0,0,0,1,1,1],
                    [-1,0,1,-1,0,1,-1,0,1]
                  ])
        elif np.all(box_size==np.array([2,3,3])):
            Box = np.array([
                    [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
                    [-1,-1,-1,0,0,0,1,1,1,-1,-1,-1,0,0,0,1,1,1],
                    [-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1,-1,0,1]
                  ])
        elif np.all(box_size==np.array([1,1,1])):
            Box = np.array([[0],[0],[0]])
        elif np.all(box_size==np.array([3,3])):
            Box = np.array([
                    [-1,-1,-1,0,0,0,1,1,1],
                    [-1,0,1,-1,0,1,-1,0,1]
                  ])
        else:
            print("Box not implemented. see pp.signal._generate_box_indices")
            quit()
    else:
        nElements = Box.shape[1]
        
        
    '''#It was
    Centers_rep = np.repeat(Centers,nElements,axis=0).T.reshape((nCenters,nCoord,nElements))
    for coord in np.arange(nCoord):
        #This was the original one, but it's something weird (compare it with the reshaped shape above
        #Centers_rep[coord] += Box[coord]
    Indices = Centers_rep.reshape((nCoord,nCenters*nElements))
    '''
    
    Centers_rep = np.repeat(Centers,nElements,axis=0).T.reshape((nCoord,nCenters,nElements))
    
    for coord in np.arange(nCoord):
        Centers_rep += Box[:,None,:]
    
    Indices = Centers_rep.reshape((nCoord,nCenters*nElements))
    for q in np.arange(len(shape)):
        np.clip(Indices[q],0,shape[q]-1,Indices[q])
    
    return Indices
    
def _slice_array(Array, Indices):
    if len(Array.shape)==2:
        return Array[Indices[0],Indices[1]]
    elif len(Array.shape)==3:
        return Array[Indices[0],Indices[1],Indices[2]]
    
def extract(Frames, Neurons, method="box", framePixelOffset=0, **kwargs):
    '''Extracts the intensities in Frames at the position specified by Neurons.
    
    Parameters
    ----------
    Frames: numpy array
        Frames[z,y,x] images. Pass an already-sliced array if there are multiple
        channels. E.g., if you have Frames[z,ch,y,x], pass Frames[:,ch,...]
    Neurons: numpy array
        Neurons[n] zyx position at which to extract the intensity for neuron n.
    method: string, optional
        Method used to extract the intensities. Default: box.
    framePixelOffset: float, optional
        Offset of the pixel intensities possibly set by the camera. For the 
        Hamamatsu ORCA Flash it is approximately 100 for each pixel in the bin.
        Subtracted only here to avoid problems with integer arithmetic or 
        unnecessary conversions to float. Default: 0
    
    Returns
    -------
    Signal: numpy array
        Signal[n] gives the intensity of the neuron n.
    '''
    nNeuron = Neurons.shape[0]
    nCoord = Neurons.shape[1]
    
    if method=="box":
        box_size = kwargs['box_size']
        nElements = np.prod(box_size)
        
        try: select_max = kwargs['select_max']
        except: select_max = False
        
        try: select_max_n = kwargs['select_max_n']
        except: select_max_n = 5
        
        if len(box_size)!=nCoord:
            print("Number of coordinates in Neuron coordinate and box_size "+\
                "don't match.")
            quit()

        # Generate indices for all the pixels in the boxes surrounding each neuron
        Indices = _generate_box_indices(Neurons, box_size, shape=Frames.shape)
        
        # Extract the values for each pixel
        Values = _slice_array(Frames,Indices).astype(float)-framePixelOffset
        
        # Reshape and sum values in each box
        Values = Values.reshape((nNeuron,nElements))
        np.clip(Values,0,None,Values)
        if not select_max:
            Signal = np.average(Values, axis=1)
        else:
            Signal = np.average(np.sort(Values,axis=1)[:,-select_max_n:],axis=1)
    
        return Signal
    elif method=="weightedMask":
        weights = kwargs['weights']
        
        #hard coded indices - curvature extraction in neuronsegmentation-c
        nElements = 51
        Box = np.array([
            [-3,-2,-2,-2,-2,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3],
            [0, -1,0,0,0,1, -2,-1,-1,-1,0,  0,0,0,0, 1,1,1,2, -2,-1,-1,-1,0,  0,0,0,0, 1,1,1,2, -2,-1,-1,-1,0,  0,0,0,0, 1,1,1,2, -1,0,0,0,1, 0],
            [0,  0,1,0,1,0,  0,-1, 0, 1,-2,-1,0,1,2,-1,0,1,0,  0,-1, 0, 1,-2,-1,0,1,2,-1,0,1,0,  0,-1, 0, 1,-2,-1,0,1,2,-1,0,1,0,  0,1,0,1,0, 0]
            ])
            
        Box = []
        iz = -3
        Box.append([iz,-1,-1])
        Box.append([iz,-1,0])
        Box.append([iz,-1,1])
        Box.append([iz,0,-1])
        Box.append([iz,0,0])
        Box.append([iz,0,1])
        Box.append([iz,1,-1])
        Box.append([iz,1,0])
        Box.append([iz,1,0])
        
        iz = -2
        Box.append([iz,-2,0])
        Box.append([iz,-1,-1])
        Box.append([iz,-1,0])
        Box.append([iz,-1,1])
        Box.append([iz,0,-2])
        Box.append([iz,0,-1])
        Box.append([iz,0,0])
        Box.append([iz,0,1])
        Box.append([iz,0,2])
        Box.append([iz,1,-1])
        Box.append([iz,1,0])
        Box.append([iz,1,1])
        Box.append([iz,2,0])
            
        for iz in np.array([-1,0,1]):
            Box.append([iz,-3,0])
            Box.append([iz,-2,-1])
            Box.append([iz,-2,0])
            Box.append([iz,-2,1])
            Box.append([iz,-1,-2])
            Box.append([iz,-1,-1])
            Box.append([iz,-1,0])
            Box.append([iz,-1,1])
            Box.append([iz,-1,2])
            Box.append([iz,0,-3])
            Box.append([iz,0,-2])
            Box.append([iz,0,-1])
            Box.append([iz,0,0])
            Box.append([iz,0,1])
            Box.append([iz,0,2])
            Box.append([iz,0,3])
            Box.append([iz,1,-2])
            Box.append([iz,1,-1])
            Box.append([iz,1,0])
            Box.append([iz,1,1])
            Box.append([iz,1,2])
            Box.append([iz,2,-1])
            Box.append([iz,2,0])
            Box.append([iz,2,1])
            Box.append([iz,3,0])
            
        iz = 2
        Box.append([iz,-2,0])
        Box.append([iz,-1,-1])
        Box.append([iz,-1,0])
        Box.append([iz,-1,1])
        Box.append([iz,0,-2])
        Box.append([iz,0,-1])
        Box.append([iz,0,0])
        Box.append([iz,0,1])
        Box.append([iz,0,2])
        Box.append([iz,1,-1])
        Box.append([iz,1,0])
        Box.append([iz,1,1])
        Box.append([iz,2,0])
        
        iz = 3
        Box.append([iz,-1,-1])
        Box.append([iz,-1,0])
        Box.append([iz,-1,1])
        Box.append([iz,0,-1])
        Box.append([iz,0,0])
        Box.append([iz,0,1])
        Box.append([iz,1,-1])
        Box.append([iz,1,0])
        Box.append([iz,1,0])
        
        Box = np.array(Box).T
        
        TM = np.zeros((119,51))
        TM[0:9,0] = 1.0
        TM[-9:,-1] = 1.0
        
        for sab in [[9,1],[9+13+25+25+25,1+5+13+13+13]]:
            sa = sab[0]
            sb = sab[1]
            TM[sa+0,sb+0] = 1.0
            TM[sa+1,sb+0] = TM[sa+1,sb+1] = TM[sa+1,sb+2] = 1./3.
            TM[sa+2,sb+0] = TM[sa+2,sb+2] = 0.5
            TM[sa+3,sb+0] = TM[sa+3,sb+2] = TM[sa+3,sb+3] = 1./3.0
            TM[sa+4,sb+1] = 1.0
            TM[sa+5,sb+1] = TM[sa+5,sb+2] = 0.5
            TM[sa+6,sb+2] = 1.0
            TM[sa+7,sb+2] = TM[sa+7,sb+3] = 0.5
            TM[sa+8,sb+3] = 1.0
            TM[sa+9,sb+1] = TM[sa+9,sb+2] = TM[sa+9,sb+4] = 1./3.
            TM[sa+10,sb+2] = TM[sa+10,sb+4] = 0.5
            TM[sa+11,sb+3] = TM[sa+11,sb+2] = TM[sa+11,sb+4] = 1./3.
            TM[sa+12,sb+4] = 1.0
        
        for sab in [[9+13,1+5],[9+13+25,1+5+13],[9+13+25+25,1+5+13+13]]:
            sa = sab[0]
            sb = sab[1]
            TM[sa+0,sb+0] = 1.0
            TM[sa+1,sb+0] = TM[sa+1,sb+1] = TM[sa+1,sb+2] = 1./3.
            TM[sa+2,sb+0] = TM[sa+2,sb+2] = 0.5
            TM[sa+3,sb+0] = TM[sa+3,sb+2] = TM[sa+3,sb+3] = 1./3.
            TM[sa+4,sb+1] = TM[sa+4,sb+4] = TM[sa+4,sb+5] = 1./3.
            TM[sa+5,sb+1] = TM[sa+5,sb+2] = TM[sa+5,sb+5] = TM[sa+5,sb+6] = 0.25
            TM[sa+6,sb+2] = TM[sa+6,sb+6] = 0.5
            TM[sa+7,sb+2] = TM[sa+7,sb+3] = TM[sa+7,sb+6] = TM[sa+7,sb+7] = 0.25
            TM[sa+8,sb+3] = TM[sa+8,sb+7] = TM[sa+8,sb+8] = 1./3.
            TM[sa+9,sb+4] = 1.0
            TM[sa+10,sb+4] = TM[sa+10,sb+5] = 0.5
            TM[sa+11,sb+5] = TM[sa+11,sb+6] = 0.5
            TM[sa+12,sb+6] = 1.0
            TM[sa+13,sb+6] = TM[sa+13,sb+7] = 0.5
            TM[sa+14,sb+7] = TM[sa+14,sb+8] = 0.5
            TM[sa+15,sb+8] = 1.0
            TM[sa+16,sb+4] = TM[sa+16,sb+5] = TM[sa+16,sb+9] = 1./3.
            TM[sa+17,sb+5] = TM[sa+17,sb+6] = TM[sa+17,sb+9] = TM[sa+17,sb+10] = 0.25
            TM[sa+18,sb+6] = TM[sa+17,sb+10] = 0.5
            TM[sa+19,sb+6] = TM[sa+19,sb+7] = TM[sa+19,sb+10] = TM[sa+19,sb+11] = 1./3.
            TM[sa+20,sb+7] = TM[sa+20,sb+8] = TM[sa+20,sb+11] = 1./3.
            TM[sa+21,sb+9] = TM[sa+21,sb+10] = TM[sa+21,sb+12] = 1./3.
            TM[sa+22,sb+10] = TM[sa+22,sb+12] = 0.5
            TM[sa+23,sb+10] = TM[sa+23,sb+11] = TM[sa+23,sb+12] = 1./3.
            TM[sa+24,sb+12] = 1.0
        
        weights_new = np.zeros((weights.shape[0],119))
        for qu in np.arange(weights.shape[0]):
            weights_new[qu] = np.dot(TM,weights[qu])
        
        weights = weights_new
        nElements = 119
        
        # Generate indices based on box
        Indices = _generate_box_indices(Neurons, Box=Box, shape=Frames.shape)
        
        # Extract the values for each pixel
        Values = _slice_array(Frames,Indices).astype(float)
        
        # Reshape, multiply by weights and sum values in each box
        Values = Values.reshape((nNeuron,nElements))-framePixelOffset
        np.clip(Values,0,None,Values)
        wValues = Values*weights
        wValues /= np.sum(weights,axis=1)[:,None]
        
        try:
            Signal = np.average(Values, axis=1)
        except:
            print(Neurons)
            print(Indices)
            print(Values)
        
        return Signal
        
    elif method=="weightedMask2d":
        weights = kwargs['weights']
        
        #hard coded indices - curvature extraction in neuronsegmentation-c
        nElements = 13
        Box = np.array([
            [0,0,0,0,0,0,0,0,0,0,0,0,0],
            [-2,-1,-1,-1,0,  0,0,0,0, 1,1,1,2],
            [0,-1, 0, 1,-2,-1,0,1,2,-1,0,1,0]
            ])
        
        # Generate indices based on box
        Indices = _generate_box_indices(Neurons, Box=Box, shape=Frames.shape)
        
        # Extract the values for each pixel
        Values = _slice_array(Frames,Indices).astype(float)
        
        # Reshape, multiply by weights and sum values in each box
        Values = Values.reshape((nNeuron,nElements))-framePixelOffset
        wValues = Values*weights
        wValues /= np.sum(weights,axis=1)[:,None]
        
        Signal = np.average(Values, axis=1) 
        
        return Signal
        
    elif method=="fitSphere":
        # Doesn't really seem to help. Dropping it right now..
        
        # Right now this is fixed, it cannot be chosen
        box_size = (2,3,3)
        nElements = np.prod(box_size)
        
        # Generate indices based on box
        Indices = _generate_box_indices(Neurons, box_size)
        
        # Extract the values for each pixel, reshape, subtract camera offset
        Values = _slice_array(Frames,Indices).astype(float)
        Values = Values.reshape((nNeuron,nElements))-framePixelOffset
        
        # Calculate average signal at central plane and next one, to be used
        # to extract the radius of the neuron. Normalize them
        Y = np.empty((Values.shape[0],2))
        Y[:,0] = np.average(Values[:,0:9],axis=-1)
        Y[:,1] = np.average(Values[:,9:],axis=-1)
        
        Y[:,0] /= Y[:,0]
        Y[:,1] /= Y[:,1]
        
        # Calculate the radius of the neuron
        R = _sphere_radius(Y)
        
        # Extract signal and multiply by the radius
        Signal = np.average(Values[:,0:9], axis=1)*R
        
        return Signal
    
    else:
        print("Signal extraction method not implemented. See pp.signal.extract")
        quit()    
        
def _sphere_radius(y, dx=1.0):
    y0sq = y[:,0]**2
    return np.sqrt(y0sq + ((y0sq-y[:,1]**2-dx**2)/(2.0*dx))**2)
            
