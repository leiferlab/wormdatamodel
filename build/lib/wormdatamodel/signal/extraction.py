import numpy as np

def _generate_box_indices(Centers, box_size=(1,3,3), Box=np.array([])):
    '''
    Generate indices for slicing of an array to extract boxes/windows of given
    size centered on Centers.
    
    Parameters
    ----------
    Centers: np.array of integers
        Centers[center_index, coordinate]. Coordinates are in slicing order, not
        plotting order (i.e. z,y,x for volumetric images).
    
    box_size: list of odd integers
        Size of the boxes centered on each Center[i]. Again, coordinates are in
        slicing order.
    '''
    
    nCenters = Centers.shape[0]
    nCoord = Centers.shape[1]
    
    if Box.shape[0]==0:
        box_size = np.array(box_size)
        nElements = np.prod(box_size)
        if np.all(box_size==np.array([3,5,5])):
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
        
    Centers_rep = np.repeat(Centers,nElements,axis=0).T.reshape((nCenters,nCoord,nElements))
    for coord in np.arange(nCoord):
        Centers_rep[coord] += Box[coord]
    
    Indices = Centers_rep.reshape((nCoord,nCenters*nElements))
    
    return Indices
    
def _slice_array(Array, Indices):
    if len(Array.shape)==2:
        return Array[Indices[0],Indices[1]]
    elif len(Array.shape)==3:
        return Array[Indices[0],Indices[1],Indices[2]]
    
def extract(Frames, Neurons, method="box", framePixelOffset=0, **kwargs):
    nNeuron = Neurons.shape[0]
    nCoord = Neurons.shape[1]
    
    if method=="box":
        box_size = kwargs['box_size']
        nElements = np.prod(box_size)
        
        if len(box_size)!=nCoord:
            print("Number of coordinates in Neuron coordinate and box_size don't \
                match.")
            quit()

        # Generate indices for all the pixels in the boxes surrounding each neuron
        Indices = _generate_box_indices(Neurons, box_size)
        
        # Extract the values for each pixel
        Values = _slice_array(Frames,Indices).astype(np.float)-framePixelOffset
        
        # Reshape and sum values in each box
        Values = Values.reshape((nNeuron,nElements))
        Signal = np.average(Values, axis=1)
    
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
        
        # Generate indices based on box
        Indices = _generate_box_indices(Neurons, Box=Box)
        
        # Extract the values for each pixel
        Values = _slice_array(Frames,Indices).astype(np.float)
        
        # Reshape, multiply by weights and sum values in each box
        Values = Values.reshape((nNeuron,nElements))-framePixelOffset
        wValues = Values*weights
        wValues /= np.sum(weights,axis=1)[:,None]
        
        Signal = np.average(Values, axis=1)
        
        return Signal
        
    elif method=="fitSphere":
        # Right now this is fixed, it cannot be chosen
        box_size = (2,3,3)
        nElements = np.prod(box_size)
        
        # Generate indices based on box
        Indices = _generate_box_indices(Neurons, box_size)
        
        # Extract the values for each pixel, reshape, subtract camera offset
        Values = _slice_array(Frames,Indices).astype(np.float)
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
            
