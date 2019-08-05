#!/usr/bin/env python

from distutils.core import setup

setup(name='wormdatamodel',
      version='1.0',
      description='Data models for whole brain imaging recordings.',
      author='Francesco Randi',
      author_email='francesco.randi@gmail.com',
      packages=['wormdatamodel','wormdatamodel.data','wormdatamodel.signal'],
     )
