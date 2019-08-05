import numpy as np
import pygmmreg as pyg

filename = "/tigress/LEIFER/francesco/pumpprobe/immobilizationtest/REDGREEN_objectivesregistration_20190717_094840/"

def redToGreen(Cervelli_R, filename=filename):
    '''
    Transforms coordinates from the red to the green part of the image. The
    transformation is purely 2D, the z coordinates are untouched.
    
    Parameters
    ----------
    Cervelli_R: numpy array (or irrarray or equivalent)
        Cervelli_R[j,n] gives the coordinate n of point j. The coordinates are
        in zyx (indexing) ordering.
        
    Returns
    -------
    Cervelli_G: numpy array (or irrarray or equivalent)
        Analogous to Cervelli_R, with y and x coordinates transformed.
    '''

    ## Load tps transformation pre-computed from LabView
    Ctrl, Param, normParam, denormParam = pyg.loadParam(filename, source="LabView", filenamebase="redGreenRegistration")
    
    ## Get an array [[x,y,-1],...] to do the 2D transformation
    # The -1 comes from the fact that the transformation was fitted in LabView
    # with points meant to go in the display vi, which has all:-1 for z.

    # Deep copy and create new neurons class for the transformation
    #Cervelli_G = sb.neurons(np.copy(Cervelli.coord),volFrame0)
    Cervelli_G = Cervelli_R.copy()

    # Flip zyx to xyz, and set all the z to -1 for the 2D transformation
    # Copy just coord to avoid passing weird stuff to the C code
    Points = np.copy(Cervelli_G.coord[:,::-1].astype(np.float)) 
    Points[:,0:2] = Points[:,0:2]
    Points[:,2] = -1.0
        
    ## Do the 2D transformation
    PointsB = pyg.transformation(Points, Ctrl, Param, normParam, denormParam)
    
    # Copy the x y coordinates back into the neurons object Cervelli_G.
    # These will be directly the indices you use to extract the signal from the
    # frames.
    Cervelli_G[:,1:] = np.rint(PointsB[:,0:2][:,::-1]).astype(np.int)
    
    return Cervelli_G
