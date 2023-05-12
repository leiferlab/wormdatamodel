#!/usr/bin/env python
'''
Contains functions related to the mapping between the red and the green
channels of the frames. Its functions are made available in the 
wormdatamodel.data namespace.

Imports
-------
numpy, scipy.io, pygmmreg, matplotlib.pyplot
'''

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

foldername = "/tigress/LEIFER/francesco/pumpprobe/immobilizationtest/REDGREEN_objectivesregistration_20190717_094840/"

def redToGreen(Cervelli_R, source="LabView", folder=foldername):
    '''Transforms coordinates from the red to the green part of the image. The
    transformation is purely 2D, the z coordinates are untouched.
    
    Parameters
    ----------
    Cervelli_R: numpy array (or irrarray or equivalent)
        Cervelli_R[j,n] gives the coordinate n of point j. The coordinates are
        in zyx (indexing) ordering.
    source: string
        Source from where the geometric transformation parameters have been
        generated (up to now, all the sources produce the same format as 
        "LabView").
    folder: string
        Folder containing the parameters of the geometric transformation.
        
    Returns
    -------
    Cervelli_G: numpy array (or irrarray or equivalent)
        Analogous to Cervelli_R, with y and x coordinates transformed.
    '''
    
    import pygmmreg as pyg

    ## Load tps transformation pre-computed from LabView
    Ctrl, Param, normParam, denormParam = pyg.loadParam(
            folder, source=source, filenamebase="redGreenRegistration")

    ## Get an array [[x,y,-1],...] to do the 2D transformation
    # The -1 comes from the fact that the transformation was fitted in LabView
    # with points meant to go in the display vi, which has all:-1 for z.

    # Deep copy and create new neurons class for the transformation
    #Cervelli_G = sb.neurons(np.copy(Cervelli.coord),volFrame0)
    Cervelli_G = Cervelli_R.copy()

    # Flip zyx to xyz, and set all the z to -1 for the 2D transformation
    # Copy just coord to avoid passing weird stuff to the C code
    Points = np.copy(Cervelli_G[:,::-1].astype(float)) 
    Points[:,0:2] = Points[:,0:2]
    Points[:,2] = -1.0

    ## Do the 2D transformation
    PointsB = pyg.transformation(Points, Ctrl, Param, normParam, denormParam)
    
    # Copy the x y coordinates back into the neurons object Cervelli_G.
    # These will be directly the indices you use to extract the signal from the
    # frames.
    Cervelli_G[:,1:] = np.rint(PointsB[:,0:2][:,::-1]).astype(int)
    
    return Cervelli_G
    
def genRedToGreen(folder, plot=False):
    '''Generates the files containing the parameters and the control points
    defining the transformation from red to green, starting from the 
    alignments.mat file from the old pipeline.
    
    Note: In the .mat file, the first array loaded has to be the Scene, the 
    second the Model. XY are inverted.
    
    Parameters
    ----------
    folder: string
        Folder containing the alignments.mat file.
        
    Saves to file the results. Use directly redToGreen, passing the same folder.
    '''
    
    import pygmmreg as pyg
    
    if folder[-1]!="/":folder+="/"
    cont=sio.loadmat(folder+"alignments.mat")
    
    '''
    A = cont['alignments'][0][0][1][0][0][2]
    B = cont['alignments'][0][0][1][0][0][3]
    '''
    B = cont['alignments'][0][0][1][0][0][2]
    A = cont['alignments'][0][0][1][0][0][3]
    
    Scale = np.array([0.8,0.5,0.3])
    nscale = len(Scale)
    L = np.zeros_like(Scale)

    Model = np.ones((A.shape[0],A.shape[1]+1))*(-1.0)
    Scene = np.ones((B.shape[0],B.shape[1]+1))*(-1.0)
    Model[:,0:2] = A[:,::-1]
    Model[:,2] = -1.0
    Scene[:,0:2] = B[:,::-1]
    Scene[:,2] = -1.0
    Ctrl = np.copy(Model)

    M = Model.shape[0]
    D = Model.shape[1]
    S = Scene.shape[0]
    N = Ctrl.shape[0]

    Param = np.zeros((N,D),dtype=np.float64)
    Param[1,0] = 1.
    Param[2,1] = 1.

    normParams = np.zeros(4)
    denormParams = np.zeros(4)

    Out = np.zeros((M,D))
    pyg.register(D,Param,L,nscale,Scale,N,Ctrl,M,Model,S,Scene,M,Out,normParams,denormParams)
    pyg.saveParam(folder,Ctrl,Param,normParams,denormParams)
    
    if plot:
        plt.plot(Out.T[0],Out.T[1],'o')
        plt.plot(Scene.T[0],Scene.T[1],'o')
        plt.show()
