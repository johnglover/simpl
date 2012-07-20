import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

cimport base

np.import_array()

cdef class Peak:
    def __cinit__(self, create_new=True):
        if create_new:
            self.thisptr = new c_Peak()
            self.created = True
        else:
            self.created = False

    def __dealloc__(self):
        if self.created:
            del self.thisptr

    cdef set_peak(self, c_Peak* p):
        self.thisptr = p

    property amplitude:
        def __get__(self): return self.thisptr.amplitude
        def __set__(self, double x): self.thisptr.amplitude = x

    property frequency:
        def __get__(self): return self.thisptr.frequency
        def __set__(self, double x): self.thisptr.frequency = x

    property phase:
        def __get__(self): return self.thisptr.phase
        def __set__(self, double x): self.thisptr.phase = x


cdef class Frame:
    def __cinit__(self, size=None, create_new=True, alloc_memory=False):
        if create_new:
            if size:
                self.thisptr = new c_Frame(size, alloc_memory)
            else:
                self.thisptr = new c_Frame()
            self.created = True
        else:
            self.created = False

    def __dealloc__(self):
        if self.created and self.thisptr:
            del self.thisptr
            self.thisptr = <c_Frame*>0

    cdef set_frame(self, c_Frame* f):
        self.thisptr = f

    # peaks
    property num_peaks:
        def __get__(self): return self.thisptr.num_peaks()
        def __set__(self, int i): self.thisptr.num_peaks(i)

    property max_peaks:
        def __get__(self): return self.thisptr.max_peaks()
        def __set__(self, int i): self.thisptr.max_peaks(i)

    def add_peak(self, Peak p not None):
        self.thisptr.add_peak(p.thisptr)

    def add_peaks(self, peaks not None):
        for p in peaks:
            self.add_peak(p)

    def peak(self, int i):
        cdef c_Peak* c_p = self.thisptr.peak(i)
        p = Peak(False)
        p.set_peak(c_p)
        return p

    property peaks:
        def __get__(self):
            return [self.peak(i) for i in range(self.thisptr.num_peaks())]
        def __set__(self, peaks):
            self.add_peaks(peaks)

    def clear(self):
        self.thisptr.clear()

    # partials
    property num_partials:
        def __get__(self): return self.thisptr.num_partials()
        def __set__(self, int i): raise Exception("NotImplemented")

    property max_partials:
        def __get__(self): return self.thisptr.max_partials()
        def __set__(self, int i): self.thisptr.max_partials(i)

    def add_partial(self, Peak p not None):
        self.thisptr.add_partial(p.thisptr)

    def add_partials(self, peaks not None):
        for p in peaks:
            self.add_partial(p)

    def partial(self, int i, Peak p=None):
        cdef c_Peak* c_p
        if not p:
            c_p = self.thisptr.partial(i)
            peak = Peak(False)
            peak.set_peak(c_p)
            return peak
        else:
            self.thisptr.partial(i, p.thisptr)

    property partials:
        def __get__(self):
            return [self.partial(i) for i in range(self.thisptr.num_partials())]
        def __set__(self, peaks):
            self.add_partials(peaks)

    # audio buffers
    property size:
        def __get__(self): return self.thisptr.size()
        def __set__(self, int i): self.thisptr.size(i)

    property audio:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.thisptr.size()
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.thisptr.audio())
        def __set__(self, np.ndarray[dtype_t, ndim=1] a):
            self.thisptr.audio(<double*> a.data)

    property synth:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.thisptr.size()
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.thisptr.synth())
        def __set__(self, np.ndarray[dtype_t, ndim=1] a):
            self.thisptr.synth(<double*> a.data)

    property residual:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.thisptr.size()
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.thisptr.residual())
        def __set__(self, np.ndarray[dtype_t, ndim=1] a):
            self.thisptr.residual(<double*> a.data)

    property synth_residual:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.thisptr.size()
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.thisptr.synth_residual())
        def __set__(self, np.ndarray[dtype_t, ndim=1] a):
            self.thisptr.synth_residual(<double*> a.data)
