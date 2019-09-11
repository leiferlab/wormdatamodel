import numpy as np
import mistofrutta as mf

class volume:
    
    def __init__(self, frames, z, dx=0.4, dy=0.4):
        self.frames = frames
        self.dx = dx
        self.dy = dy
        self.z = z
        
    def plot(self, wait=False):
        mf.plt.hyperstack(self.frames, order='zc', cmap='viridis', wait=wait)
