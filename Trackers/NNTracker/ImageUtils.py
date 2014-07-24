"""
Some tools for efficiently manipulating images.
Author: Travis Dick (travis.barry.dick@gmail.com)
"""

import itertools

import cv2
import numpy as np
from scipy import weave
from scipy.weave import converters

from Homography import *

def draw_region(img, corners, color, thickness=1, draw_x=False):
    """ Draws a warped rectangle in an image.

    Parameters:
    -----------
    img : (n,m) numpy array
      The image to be drawn into
      
    corners : (2,4) numpy array
      An array whose columns are the corners of the warped
      rectangle. Points should be in (counter-)clockwise order.

    color : scalar or (b,g,r) triple
      The color to make the region. In black and white images
      a scalar between 0 and 1 specifies intensity, and in 
      color images, a (b,g,r) triple (0 <= b,g,r <= 255) specifies
      the color.

    thickness : integer
      The width in pixels of the lines used.
      
    draw_x : boolean
      Whether or not to connect opposite corners.
    """
    for i in xrange(4):
        p1 = (int(corners[0,i]), int(corners[1,i]))
        p2 = (int(corners[0,(i+1)%4]), int(corners[1,(i+1)%4]))
        cv2.line(img, p1, p2, color, thickness)
    if draw_x:
        for i in xrange(4):
            p1 = (int(corners[0,i]), int(corners[1,i]))
            p2 = (int(corners[0,(i+2)%4]), int(corners[1,(i+2)%4]))
            cv2.line(img, p1, p2, color, thickness)

def to_grayscale(img):
    """ Converts an bgr8 image into a grayscale image.

    Parameters:
    -----------
    img : (n,m,d) numpy array
      The input image, with d channels. Entries should take
      values in the range {0,1,...,255}.
    
    Returns:
    --------
    An (n,m) numpy array with entries {0,1,...,255}
    """
    (height, width, depth) = img.shape
    grayscale = np.empty((height,width), dtype=np.float64)
    code = \
    """
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        double mean = 0;
        for (int k = 0; k < depth; k++) mean += img(i,j,k);
        grayscale(i,j) = mean / depth;
      }
    }
    """
    weave.inline(code, ['height', 'width', 'depth', 'grayscale', 'img'],
                 type_converters=converters.blitz,
                 compiler='gcc')
    return grayscale

def sample_region(img, pts, warp=np.eye(3), result=None):
    """ Samples the image intenisty at a collection of points.

    Notes: 
    ------
      - Only works with grayscale images. 
      - All points outside the bounds of the image have intensity 128.

    Parameters:
    -----------
    img : (n,m) numpy array
      The image to be sampled from.
     
    pts : (2,k) numpy array
      The points to be sampled out of the image. These may be sub-pixel
      coordinates, in which case bilinear interpolation is used.

    result : (k) numpy array (optional)
      Optionally you can pass in a results vector which will store the 
      sampled vector. If you do not supply one, this function will allocate
      one and return a reference.
    
    Returns:
    --------
    Returns a (k) numpy array containing the intensities of the given 
    sub-pixel coordinates in the provided image.
    """
    num_pts = pts.shape[1]
    (height, width) = img.shape
    w = np.asarray(warp)
    #import pdb;pdb.set_trace()
    if result == None: result = np.empty(num_pts)
    support_code = \
    """
    double bilinear_interp(blitz::Array<double,2> img, int width, int height, double x, double y) {
      using std::floor;
      using std::ceil;
      const int lx = floor(x);
      const int ux = ceil(x);
      const int ly = floor(y);
      const int uy = ceil(y);
      if (lx < 0 || ux >= width || ly < 0 || uy >= height) return 128;
      const double ulv = img(ly,lx);
      const double urv = img(ly,ux);
      const double lrv = img(uy,ux);
      const double llv = img(uy,lx);
      const double dx = x - lx;
      const double dy = y - ly;
      return ulv*(1-dx)*(1-dy) + urv*dx*(1-dy) + llv*(1-dx)*dy + lrv*dx*dy;
    }
    """
    code = \
    """
    for (int i = 0; i < num_pts; i++) {
      double d = w(2,0)*pts(0,i) + w(2,1)*pts(1,i) + w(2,2);
      double x = (w(0,0)*pts(0,i) + w(0,1)*pts(1,i) + w(0,2)) / d;
      double y = (w(1,0)*pts(0,i) + w(1,1)*pts(1,i) + w(1,2)) / d;
      result(i) = bilinear_interp(img, width, height, x, y);
    }
    """
    weave.inline(code, ["img", "result", "w", "pts", "num_pts", "width", "height"],
                 support_code=support_code, headers=["<cmath>"],
                 type_converters=converters.blitz,
                 compiler='gcc')
    return result

def sample_and_normalize(img, pts, warp=np.eye(3)):
    """ Samples the image intensity at a collection of points 
    and normalizes the result.

    Identical to sample_region, except the result is shifted
    so that it's components have mean 0.

    See Also:
    ---------
    sample_region
    """
    result = sample_region(img, pts, warp);
    #result -= result.mean()
    return result

def image_gradient(img, pts, warp=None):
    """ Computes the spatial image gradient at a collection of points.

    Parameters:
    -----------
    img : (n,m) numpy array
      The image whose gradient is to be computed
    
    pts : (2,k) numpy array
      An array where each column contains the pixel coordinates of a point.

    warp : (3,3) numpy array (Homography)
      A warp to apply to the points before samping them from the image. This
      is a change of coordinates from the template coordinate frame into the
      pixel image coordinate frame. If no warp is specified, then we assume
      that the template coordinate frame is the centered unit square.

    Returns:
    --------
    A (k,2) numpy array where the ith row is the gradient [dI/dx, dI/dy] 
    sampled at pts[:,i].
    """
    
    if warp == None:
        (height, width) = img.shape
        warp = np.matrix([[width,0,0],[0,height,0],[0,0,1]]) * \
            np.array([[1,0,0.5], [0,1,0.5], [0,0,1]])

    pts = apply_to_pts(warp, pts)

    xo = np.array([1,0]).reshape(-1,1)
    yo = np.array([0,1]).reshape(-1,1)

    dx = (sample_region(img, pts + xo) - sample_region(img, pts - xo))/2
    dy = (sample_region(img, pts + yo) - sample_region(img, pts - yo))/2
    
    return np.array([dx,dy]).T

def res_to_pts(res, ul=(-.5,-.5), lr=(.5,.5)):
    """ Makes an array of evenly spaced points in the centered unit square.

    Parameters:
    -----------
    res : (width integer, height integer)
      width is the number of points horizontally and height is the number
      of points vertically.

    Returns:
    --------
    A (2, width*height) numpy array where each column is one of the grid points.
    """
    return np.array(list(map(lambda (x,y):(y,x), 
                             itertools.product(np.linspace(ul[0], lr[0], res[0]), 
                                               np.linspace(ul[1], lr[1], res[1]))))).T
