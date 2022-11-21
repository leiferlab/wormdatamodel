'''
Functions related to saving and loading of signal arrays to and from files.
The numpy savetxt and loadtxt are used, in addition to a header containing the
json-serialized information about how the signal was extracted.

Imports
-------
numpy, wormdatamodel, pkg_resources, json
'''

import numpy as np
import wormdatamodel as wormdm
import pkg_resources
import json
import os
import pickle
import re

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
    header_string = json.dumps(header)
    
    # Save plain text file for compatibility
    np.savetxt(folder+filename, Signal, header=header_string)
    
    # Save pickled file for faster processing
    #data = {"signal": Signal, "header": header}
    #pickle_filename = ".".join([filename.split(".")[0],"pickle"])
    #pickle_file = open(folder+pickle_filename,"wb")
    #pickle.dump(data,pickle_file)
    #pickle_file.close()
    
    
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
    #pickle_filename = ".".join([filename.split(".")[0],"pickle"])
    
    # If available, load pickled file
    # Otherwise, load from txt file and create pickle cache
    #if os.path.isfile(folder+pickle_filename):
    #    f = open(folder+pickle_filename,"rb")
    #    data = pickle.load(f)
    #    f.close()
    #    Signal = data["signal"]
    #    info = data["header"]        
    #else:
    f = open(folder+filename,"r")
    l = f.readline()
    f.close()
    try:
        info = json.loads(l[2:])
    except:
        info = {}
    
    Signal = np.loadtxt(folder+filename)
    
    #data = {"signal": Signal, "header": info}
    #pickle_file = open(folder+pickle_filename,"wb")
    #pickle.dump(data,pickle_file)
    #pickle_file.close()
    
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
    
def manually_added_neurons_n(folder,added_neurons_fname="added_ref_neurons.txt"):
    '''Returns the number of neurons that have been manually added. They are the
    last neurons in the list.
    
    Parameters
    ----------
    folder: string
        Folder of the dataset.
    
    Returns
    -------
    n: int
        Number of neurons that have been manually added.
    '''
    if folder[-1]!="/": folder+="/"
    
    n = 0
    if os.path.isfile(folder+added_neurons_fname):
        an = np.loadtxt(folder+added_neurons_fname)
        n = an.shape[0]
        
    return n
    
def load_ds_list(fname,tags=None,exclude_tags=None,return_tags=False):
    '''Loads the list of dataset folder names given the filename of a 
    text file containing a folder name per line. Comments start with # (both
    for whole line and for annotations after the folder name).
    
    Parameters
    ----------
    fname: str
        Name of the txt file containing the list of datasets.
    tags: str (optional)
        Space-separated tags to select datasets. Default: None.
    exclude_tags: str (optional)
        Space-separated tags to exclude in the dataset selection. Default:
        None.
    
    Returns
    -------
    ds_list: list of str
        List of the dataset folder names.
    '''
    
    f = open(fname,"r")
    ds_list = []
    ds_tags_lists = []
    if tags is not None: tags = tags.split(" ")
    if exclude_tags is not None: exclude_tags = exclude_tags.split(" ")
    
    for l in f.readlines():
        if l[0] not in ["#","\n"]: # Ignore commented lines
            # Remove commented annotations
            l2 = l.split("#")[0]
            # Get tags
            if len(l.split("#"))==0: 
                # There are no tags
                continue
            tgs = l.split("#")[1].split(" ")
            for it in np.arange(len(tgs)):
                tgs[it] = re.sub(" ","",tgs[it])
                tgs[it] = re.sub("\n","",tgs[it])
                tgs[it] = re.sub("\r","",tgs[it])
                tgs[it] = re.sub("\t","",tgs[it])
            if tags is not None:
                ok = [t in tgs or t=="" for t in tags]
                if not np.all(ok): continue
            if exclude_tags is not None:
                not_ok = [t in tgs and t!="" for t in exclude_tags]
                if np.any(not_ok): continue
            
            # Remove blank spaces, newlines, and tabs
            l2 = re.sub(" ","",l2)
            l2 = re.sub("\n","",l2)
            l2 = re.sub("\r","",l2)
            l2 = re.sub("\t","",l2)
            
            # Complete folder path with last / if necessary
            if l2[-1]!="/":l2+="/" 
            
            ds_tags_lists.append(tgs)
            ds_list.append(l2)
    f.close()
    
    if return_tags:
        return ds_list, ds_tags_lists
    else:
        return ds_list
