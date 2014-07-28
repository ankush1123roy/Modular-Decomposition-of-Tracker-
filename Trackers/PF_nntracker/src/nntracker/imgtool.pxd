cdef double[:,:] warpimg(double[:,:] img, double[:,:] p, double[:,:] sz)
cdef double[:] weight_eval(double[:,:] samples, double[:,:] template, double[:,:] sz,double alpha)
cdef double[:] est_weight_warp(double[:,:] img, double[:,:] template, double[:,:] param, int n_samples, double[:,:] sz_T, double[:,:] affsig, double alpha)
cdef double[:,:] whiten(double[:,:] In)
