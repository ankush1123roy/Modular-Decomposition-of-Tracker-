from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

setup(
    name = "NNTracker Library",
    cmdclass = {'build_ext': build_ext},
    include_dirs = [np.get_include()],
    ext_modules = [
        Extension("nntracker.utility", ["nntracker/utility.pyx"]),
        Extension("nntracker.imgtool", ["nntracker/imgtool.pyx"],include_dirs=['nntracker']),
        Extension("nntracker.trackers.PFTracker", ["nntracker/trackers/PFTracker.pyx"]),
        Extension("nntracker.trackers.ESMTracker", ["nntracker/trackers/ESMTracker.pyx"]),
        Extension("nntracker.trackers.BMICTracker", ["nntracker/trackers/BMICTracker.pyx"]),
        Extension("nntracker.trackers.NNTracker", ["nntracker/trackers/NNTracker.pyx"])
        ]
)
    
