"""
Utilities for implementing the Sum of Conditional Variances as
described by Richa et. al.

R. Richa, R. Sznitman, R. Taylor, and G. Hager, "Visual tracking
using the sum of conditional variance," Intelligent Robots and
Systems (IROS), 2011 IEEE/RSJ International Conference on, pp.
2953-2958, 2011.

Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import numpy as np
from scipy import weave
from scipy.weave import converters

from ImageUtils import *

def scv_intensity_map(src, dst):
    conditional_probability = np.zeros((256,256))
    intensity_map = np.arange(256, dtype=np.float64)
    n = len(src)
    code = \
    """
    for (int k = 0; k < n; k++) {
      int i = int(src(k));
      int j = int(dst(k));
      conditional_probability(i,j) += 1;
    }
    for (int i = 0; i < 256; i++) {
      double normalizer = 0;
      double total = 0;
      for (int j = 0; j < 256; j++) {
        total += j * conditional_probability(i,j);
        normalizer += conditional_probability(i,j);
      }
      if (normalizer > 0) {
        intensity_map(i) = total / normalizer;
      }
    }
    """
    weave.inline(code, ['conditional_probability', 'intensity_map', 'n', 'src', 'dst'],
                 type_converters=converters.blitz,
                 compiler='gcc')
    return intensity_map

def scv_expectation(original, intensity_map):
    return intensity_map[np.floor(original).astype(np.int)]
    
