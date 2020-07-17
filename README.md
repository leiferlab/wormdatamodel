# wormdatamodel
Module implementing a data model for the whole brain recordings (more broadly, of any multichannel volumetric recording - sequence of "hyperstacks"). While there are only a few options to efficiently store an uncompressed sequence of images, so that the issues with uncompatibility of that data with different softwares are limited, metadata associated with it (z position of the image, index of the volume to which it belongs, ...) can be stored in different ways.

Data from any source can be fed in the rest of the libraries and modules, provided that an interface exposing the same methods and variables as the wormbrain.data.recording class.

Below, there is an overview of the content of the module. For more extensive documentation, see docs/wormdatamodel/index.html .

## Installation
To install the module, use (with `python` an alias for python3)
```
python -m pip install . --user
```
adding `--user` at the end if you want to install it in your home folder, or do not have privileges to write files in the standard target folder. Note that there is a C++ extension: the setup.py script will take care of its compilation, but you need to have a C++ compiler installed on your system.

## Usage
The module contains two submodules: data and signal. 

### Data
The submbodule data contains
* the class recording that is the actual data model representing the data, and that has methods to load the data (images/frames) and metadata (info about the frames, like z position for volumetric recordings, index of the volume to which they belong, events like optogenetic stimulations, ...) from file. See the documentation for the recording class for the useful attributes and methods. 
* the functions that provide the geometric mapping between the pixels in the different channels. 

### Signal
The submodule signal contains
* methods to extract the signal from the frames given the positions of the neurons (basically using advanced numpy slicing with various kind of "boxes" or "windows" centered on the neurons),
* methods to save/load the extracted signals to/from file, including metadata on how the signal has been extracted,
* the class Signal that contains the signal, preprocessing methods like nan interpolation, photobleaching correction (to do), and, based on the irrarray structure, methods that allow one to deal with irregularly spaced events (like optogenetic stimulations) with a clear syntax.

## Authors
Francesco Randi @ Leifer Lab, Princeton Physics
