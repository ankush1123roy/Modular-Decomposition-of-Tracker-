__author__ = 'Tommy'
import cv2
import numpy as np

class FilterParams:
    def __init__(self, type):
        self.type = type
        self.params = []
        self.validated = False
        self.init_success = True
        self.update = lambda: None
        self.validate = lambda: True
        if type == 'gabor':
            print 'Initializing Gabor filter'
            self.params.append(Param('ksize', 2, 1, 10, 1, type='int'))
            self.params.append(Param('sigma', 0.1, 10, 100))
            self.params.append(Param('theta', np.pi / 12, 0, 24))
            self.params.append(Param('lambd', 0.1, 10, 100, 10))
            self.params.append(Param('gamma', 0.1, 10, 100))
            self.update = lambda: cv2.getGaborKernel((self.params[0].val, self.params[0].val), self.params[1].val,
                                                     self.params[2].val, self.params[3].val, self.params[4].val)
            self.apply = lambda img: cv2.filter2D(img, -1, self.kernel)
        elif type == 'laplacian':
            print 'Initializing Laplacian filter'
            self.params.append(Param('ksize', 2, 1, 10, 1, type='int'))
            self.params.append(Param('scale', 1, 0, 10, 1, type='int'))
            self.params.append(Param('delta', 1, 0, 255, type='int'))
            self.apply = lambda img: cv2.Laplacian(img, -1, ksize=self.params[0].val,
                                                   scale=self.params[1].val, delta=self.params[2].val)
        elif type == 'sobel':
            print 'Initializing Sobel filter'
            self.params.append(Param('ksize', 2, 1, 10, 1, type='int'))
            self.params.append(Param('scale', 1, 0, 10, 1, type='int'))
            self.params.append(Param('delta', 1, 0, 255, type='int'))
            self.params.append(Param('dx', 1, 1, 5, type='int'))
            self.params.append(Param('dy', 1, 0, 5, type='int'))
            self.apply = lambda img: cv2.Sobel(img, -1, ksize=self.params[0].val, scale=self.params[1].val,
                                               delta=self.params[2].val, dx=self.params[3].val, dy=self.params[4].val)
            self.validate = lambda: validate()

            def validate():
                print 'Validating Sobel derivatives...'
                #print 'dx=',self.params[-2].val, ' dy=', self.params[-1].val
                if self.params[-1].val == 0 and self.params[-2].val == 0:
                    #print self.params[-1].name, "is ", self.params[-1].val, " while ", self.params[-2].name, " is ", self.params[-2].val
                    self.validated = False
                    return False
                if self.params[-1].val >= self.params[0].val:
                    #print self.params[0].name, "is ", self.params[0].val, " while ", self.params[-1].name, " is ", self.params[-1].val
                    self.validated = False
                    return False
                if self.params[-2].val >= self.params[0].val:
                    #print self.params[0].name, "is ", self.params[0].val, " while ", self.params[-2].name, " is ", self.params[-2].val
                    self.validated = False
                    return False
                return True
        elif type == 'scharr':
            print 'Initializing Scharr filter'
            self.params.append(Param('scale', 1, 0, 10, 1, type='int'))
            self.params.append(Param('delta', 1, 0, 255, type='int'))
            self.params.append(Param('dx', 1, 1, 1, type='int'))
            self.params.append(Param('dy', 1, 0, 1, type='int'))
            self.update = lambda: validate()
            self.apply = lambda img: cv2.Scharr(img, -1, scale=self.params[0].val,
                                                delta=self.params[1].val, dx=self.params[2].val,
                                                dy=self.params[3].val)
            self.validate = lambda: validate()

            def validate():
                print 'Validating Scharr derivatives...'
                #print 'dx=',self.params[-2].val, ' dy=', self.params[-1].val
                if self.params[-1].val == 0 and self.params[-2].val == 0:
                    self.validated = True
                    return False
                if self.params[-1].val + self.params[-2].val > 1:
                    self.validated = True
                    return False
                return True
        elif type == 'canny':
            print 'Initializing Canny Edge Detector'
            self.params.append(Param('low_thresh', 1, 20, 50, 0))
            self.params.append(Param('ratio', 1, 4, 10, 0))
            self.apply = lambda img: cv2.Canny(img, self.params[0].val,
                                               self.params[0].val*self.params[1].val)
        elif type == 'gauss':
            print 'Initializing Gaussian filter'
            self.params.append(Param('ksize', 2, 1, 10, 1, type='int'))
            self.params.append(Param('std', 0.1, 20, 100, 1))
            self.apply = lambda img: cv2.filter2D(img, -1, self.kernel)
            self.update = lambda: cv2.getGaussianKernel(self.params[0].val, self.params[1].val)
        elif type == 'bilateral':
            print 'Initializing Bilateral filter'
            self.params.append(Param('diameter', 1, 3, 50, 1))
            self.params.append(Param('sigma_col', 1, 3, 200, 0))
            self.params.append(Param('sigma_space', 1, 50, 200, 0))
            self.apply = lambda img: cv2.bilateralFilter(img, self.params[0].val,
                                                         self.params[1].val, self.params[2].val)
        elif type == 'median':
            print 'Initializing Median filter'
            self.params.append(Param('ksize', 2, 1, 10, 1, type='int'))
            self.apply = lambda img: cv2.medianBlur(img, self.params[0].val)
        elif type == 'DoG':
            print 'Initializing DoG filter'
            self.params.append(Param('ksize', 2, 1, 10, 1, type='int'))
            self.params.append(Param('exc_std', 0.1, 20, 100, 1))
            self.params.append(Param('inh_std', 0.1, 28, 100, 1))
            self.params.append(Param('ratio', 0.05, 50, 200))
            self.apply = lambda img: applyDoG(img)

            def applyDoG(img):
                img = img.astype(np.uint8)
                ex_img = cv2.GaussianBlur(img, (self.params[0].val, self.params[0].val),
                                          sigmaX=self.params[1].val)
                in_img = cv2.GaussianBlur(img, (self.params[0].val, self.params[0].val),
                                          sigmaX=self.params[2].val)
                dog_img = ex_img - self.params[3].val * in_img
                dog_img = dog_img.astype(np.uint8)
                return dog_img
        elif type == 'LoG':
            print 'Initializing LoG filter'
            self.params.append(Param('gauss_ksize', 2, 1, 10, 1, type='int'))
            self.params.append(Param('std', 0.1, 20, 100, 0.1))
            self.params.append(Param('lap_ksize', 2, 1, 5, 1, type='int'))
            self.params.append(Param('scale', 1, 0, 10, 1, type='int'))
            self.params.append(Param('delta', 1, 0, 255, type='int'))
            self.apply = lambda img: applyLoG(img)

            def applyLoG(img):
                gauss_img = cv2.GaussianBlur(img, (self.params[0].val, self.params[0].val),
                                             sigmaX=self.params[1].val)
                log_img = cv2.Laplacian(gauss_img, -1, ksize=self.params[2].val,
                                        scale=self.params[3].val, delta=self.params[4].val)
                log_img = log_img.astype(np.uint8)
                return log_img
        else:
            self.init_success = False
            print "Invalid filter type:", type

        self.printParamValue()
        self.kernel = self.update()
        #print 'in FilterParams __init__ kernel=', self.kernel
        print "\n" + "*" * 60 + "\n"

    def printParamValue(self):
        #print self.type, "filter:"
        for i in xrange(len(self.params)):
            print self.params[i].name, "=", self.params[i].val


class Param:
    def __init__(self, name, base, multiplier, limit, add=0.0, type='float'):
        self.name = name
        self.base = base
        self.multiplier = multiplier
        self.add = add
        self.limit = limit
        self.val = 0
        self.type=type
        self.updateValue()


    def updateValue(self, multiplier=None):
        if multiplier != None:
            self.multiplier = multiplier
        self.val = self.base * self.multiplier + self.add
        if self.type=='int':
            self.val=int(self.val)
        if multiplier != None:
            print self.name, "updated to", self.val