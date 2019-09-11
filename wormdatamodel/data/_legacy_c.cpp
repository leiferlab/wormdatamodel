//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <numpy/arrayobject.h>

// Define functions below

static PyObject *load_frames_legacy(PyObject *self, PyObject *args);

/////// Python-module-related functions and tables

// The module's method table
static PyMethodDef _legacy_cMethods[] = {
    {"load_frames_legacy", load_frames_legacy, METH_VARARGS, ""},
    {NULL, NULL, 0, NULL}
};

// The module definition function
static struct PyModuleDef _legacy_c = {
    PyModuleDef_HEAD_INIT,
    "_legacy_c",
    NULL, // Module documentation
    -1,
    _legacy_cMethods
};

// The module initialization function
PyMODINIT_FUNC PyInit__legacy_c(void) { 
        import_array(); //Numpy
        return PyModule_Create(&_legacy_c);
};

static PyObject *load_frames_legacy(PyObject *self, PyObject *args) {

    int startFrame, frameN, rowSize, rowN;
    const char *fname;
    PyObject *frames_o;
    
    if(!PyArg_ParseTuple(args, "ciiiiO", 
        &fname, &startFrame, &frameN, &rowSize, &rowN, &frames_o)) return NULL;
    
    PyObject *frames_a = PyArray_FROM_OTF(frames_o, NPY_UINT16, NPY_IN_ARRAY);
        
    // Check that the above conversion worked, otherwise decrease the reference
    // count and return NULL.                                 
    if (frames_a == NULL) {
        Py_XDECREF(frames_a);
        return NULL;
    }
    
    // Get pointers to the data in the numpy arrays.
    uint16_t *frames = (uint16_t*)PyArray_DATA(frames_a);
    
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    // Actual C code
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    
    std::ifstream file(fname, std::ios::in|std::ios::binary);
    // file.read() uses a pointer to char as destination. Declare a pointer to
    // char that will point to the array of uint16_t frames, and calculate the
    // sizes of rows and frames in chars (and not pixels, which are uint16_t).
    char *memblock;
    int framesize_char = rowSize*rowN*2;
    int rowsize_char = rowSize*2;
    int fullrowsize_char = rowSize*2*2; //(2 for full row, 2 for uint16->char)
    
    // Convert the pointer to the frames array from a pointer to uint16_t to 
    // a pointer to char, so that it can be correctly used in file.read().
    memblock = (char*)frames;
    
    // For each frame
    for(int i=0;i<frameN;i++){
        // For each full row
        for(int n=0;n<rowN;n++){
            // Copy the first half of the row (red).
            file.read(memblock, rowsize_char);
            // Move the pointer a frame (i.e. channel, half of the full frame
            // for each time) forward,
            memblock += framesize_char;
            // Read the second half row (green).
            file.read(memblock, rowsize_char);
            // Go back to the red channel.
            memblock -= framesize_char;
            // If you're not at the end of the allocated frames, move one full
            // row forward.
            if(i<(frameN-1)){ memblock += fullrowsize_char;}
        }
        // If you're not at the end of the allocated frames, move one channel 
        // forward, otherwsise you'll write the next red line on the first
        // green line of the last frame.
        if(i<(frameN-1)){memblock += framesize_char*2;}
    }
    
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    // End of C code
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    //////////////////////////////////
    
    
    // Decrease the reference count for the python objects that have been 
    // declared in this function.
    Py_XDECREF(frames_a);
    
    // Return the python object none. Its reference count has to be increased.
    Py_INCREF(Py_None);
    return Py_None;
}
