import numpy as np
import wormdatamodel as wormdm
import pkg_resources
import json

def to_file(Signal, folder, filename, method, other={}):
    if folder[-1]!="/": folder+="/"
    
    v = pkg_resources.get_distribution("wormbrain").version
    
    header = {"method": method, "version": v}
    for key in other:
        try:
            header[key] = other[key]
        except:
            pass
    header = json.dumps(header)
    
    np.savetxt(folder+filename, Signal, header=header)
    
def from_file(folder, filename):
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
    if folder[-1]!="/": folder+="/"
    
    f = open(folder+filename,"r")
    l = f.readline()
    f.close()
    try:
        info = json.loads(l[2:])
    except:
        info = {}
        
    return info
