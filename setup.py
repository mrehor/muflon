#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
import re, os, glob, subprocess

version = re.findall('__version__ = "(.*)"',
                     open('muflon/__init__.py', 'r').read())[0]

packages = [
    "muflon",
    "muflon.common",
    ]

CLASSIFIERS = """
Development Status :: 2 - Pre-Alpha
Environment :: Console
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Programming Language :: Python
Programming Language :: C++
Topic :: Scientific/Engineering :: Mathematics
"""
classifiers = CLASSIFIERS.split('\n')[1:-1]

# Collect all demo files into a list
# TODO: This is prone to omit something
demofiles = glob.glob(os.path.join("demo", "*", ".py"))
demofiles += glob.glob(os.path.join("demo", "*", "*", "*.py"))

# Collect all datafiles into a list
datafiles = glob.glob(os.path.join("data", "*"))

setup(name="MUFLON",
      version=version,
      author="Martin Řehoř",
      author_email="rehor@karlin.mff.cuni.cz",
      url="http://gitlab.karlin.mff.cuni.cz/mr-dev/muflon",
      description="MUltiphase FLOw simulatioN library",
      classifiers=classifiers,
      license="GNU LGPL v3 or later",
      packages=packages,
      package_dir={"muflon": "muflon"},
      #package_data={"muflon": ["*.h"]},
      data_files=[(os.path.join("share", "muflon", os.path.dirname(f)), [f])
                  for f in demofiles+datafiles],
    )
