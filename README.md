# wormdatamodel
Module implementing a data model for the whole brain recordings (more broadly, of any multichannel volumetric recording - sequence of "hyperstacks"). While there are only a few options to efficiently store an uncompressed sequence of images, so that the issues with uncompatibility of that data with different softwares are limited, metadata associated with it (z position of the image, index of the volume to which it belongs, ...) can be stored in different ways.

Data from any source can be fed in the rest of the libraries and modules, provided that an interface exposing the same methods and variables as the wormbrain.data.recording class.

## Installation
To install the module, use (with `python` an alias for python3)
```
python setup.py install
```
adding `--user` at the end if you want to install it in your home folder, or do not have privileges to write files in the standard target folder. Note that there is a C++ extension: the setup.py script will take care of its compilation, but you need to have a C++ compiler installed on your system.

## Usage
When creating a recording object, the class only loads the metadata of the recording and not the images/frames, so that the class can be used also in a light-weight mode that uses limited RAM. Once the object is created, you can load frames into memory with the method `load()` passing `startFrame` and `stopFrame` or, more commonly, `startVolume` and `nVolume`. Given the amount of data that you might load via this class, in complex/long scripts it might be useful to use Python contexts to create the recording object:
```
with wormdatamodel.data.recording(folder) as rec:
  rec.load(startVolume=i, nVolume=N)
```
Once loaded, the frames are accessible in the numpy array `rec.frame[z,ch,y,x]`

Among others, useful properties and methods of the class are:
- recording.volumeFirstFrame is the array with the indices of the first frames of each volume
- recording.ZZ is a list of numpy arrays with the z coordinate of the frames in each volume. 
- recording.\_get_memoryUsagePerVolume() returns an estimate of the memory generally used to store and process one volume (useful to limit the amount of data loaded to the available RAM)
- recoring.free_memory() deletes the frames.

## Authors
Francesco Randi @ Leifer Lab, Princeton Physics
