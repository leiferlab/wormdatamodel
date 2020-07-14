import numpy as np
import wormdatamodel as wormdm
import pkg_resources
import json

def to_file(Signal, folder, filename, method, other={}):
    '''Save signal array to a text file. The array is saved with np.savetxt,
    so it can be loaded directly with np.loadtxt. However, this function adds
    a commented header that contains a json serialized dictionary with 
    properties of the signal, like the method used to extract it and the version
    of this module. To retrieve and parse this header, use from_file().
    
    Parameters
    ----------
    Signal: numpy array
        2D array containing the signal.
    folder: string
        Folder in which to save the file. 
    filename: string
        Name of the destination file.
    method: string
        Method used to extract the signal from the images.
    other: dict (optional)
        Other details about the signal extraction that need to be saved.
        Default: {}.
    '''
    if folder[-1]!="/": folder+="/"
    
    #v = pkg_resources.get_distribution("wormbrain").version
    v = pkg_resources.get_distribution("wormdatamodel").version
    
    header = {"method": method, "version": v}
    for key in other:
        try:
            header[key] = other[key]
        except:
            pass
    header = json.dumps(header)
    
    np.savetxt(folder+filename, Signal, header=header)
    
def from_file(folder, filename):
    '''Loads the array containing a signal array, and retrieves and parses the
    header that contains additional information on how the signal was extracted
    from the frames.
    
    Parameters
    ----------
    folder: str
        Folder containing the file.
    filename: str
        Filename.
        
    Returns
    -------
    Signal: numpy array
        2D array containing the signal
    info: dict
        Dictionary containing the metadata of the signal.
    
    '''
    if folder[-1]!="/": folder+="/"
    
    f = open(folder+filename,"r")
    l = f.readline()
    f.close()
    try:
        info = json.loads(l[2:])
    except:
        info = {}
        
    Signal = np.loadtxt(folder+filename)
    
    return Signal, info
    
def from_file_info(folder, filename):
    '''Retrieves and parses the header that contains additional information on 
    how the signal was extracted from the frames, without loading the array
    (which can take some seconds).
    
    Parameters
    ----------
    folder: str
        Folder containing the file.
    filename: str
        Filename.
        
    Returns
    -------
    info: dict
        Dictionary containing the metadata of the signal.
    
    '''
    if folder[-1]!="/": folder+="/"
    
    f = open(folder+filename,"r")
    l = f.readline()
    f.close()
    try:
        info = json.loads(l[2:])
    except:
        info = {}
        
    return info
