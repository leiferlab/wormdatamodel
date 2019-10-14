#!/usr/bin/env python

from distutils.core import setup, Extension
import numpy

_legacy_c = Extension('wormdatamodel.data._legacy_c',
                    sources = ['wormdatamodel/data/_legacy_c.cpp'],
                    include_dirs = [
                        numpy.get_include()
                        ],
                    extra_compile_args=['-O3','-D_FILE_OFFSET_BITS=64'])#

setup(name='wormdatamodel',
      version='1.0',
      description='Data models for whole brain imaging recordings.',
      author='Francesco Randi',
      author_email='francesco.randi@gmail.com',
      packages=['wormdatamodel','wormdatamodel.data','wormdatamodel.signal'],
      ext_modules=[_legacy_c]
     )
