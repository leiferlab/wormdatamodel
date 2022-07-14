#!/usr/bin/env python

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
import numpy
import git

class CustomBuildExtCommand(build_ext):
    """build_ext command for use when numpy headers are needed."""

    def run(self):

        # Import numpy here, only when headers are needed
        import numpy
        # Add numpy headers to include_dirs
        self.include_dirs.append(numpy.get_include())
        
        # Call original build_ext command
        build_ext.run(self)

_legacy_c = Extension('wormdatamodel.data._legacy_c',
                    sources = ['wormdatamodel/data/_legacy_c.cpp'],
                    include_dirs = [],
                    extra_compile_args=['-O3','-D_FILE_OFFSET_BITS=64'])#
                    
# Get git commit info to build version number/tag
#repo = git.Repo('.git')
#git_hash = repo.head.object.hexsha
#git_url = repo.remotes.origin.url
#v = repo.git.describe(always=True)
#if repo.is_dirty(): v += ".dirty"
v = 1.5

setup(name='wormdatamodel',
      version=v,
      description='Data models for whole brain imaging recordings.',
      author='Francesco Randi',
      author_email='francesco.randi@gmail.com',
      packages=['wormdatamodel','wormdatamodel.data','wormdatamodel.signal'],
      ext_modules=[_legacy_c]
     )
