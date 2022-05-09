# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='r3meval',
    description='Policy learning from pixels using pre-trained representations, used in the R3M paper',
    install_requires=[
        'pip', 'click', 'gym==0.20.0', 'termcolor', 'mjrl', 'mj_envs', 'tabulate', 'scipy', 'transforms3d', 'moviepy'
    ],
)
