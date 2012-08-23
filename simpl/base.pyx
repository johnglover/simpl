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

    cdef copy(self, c_Peak* p):
        self.thisptr.amplitude = p.amplitude
        self.thisptr.frequency = p.frequency
        self.thisptr.phase = p.phase
        self.thisptr.bandwidth = p.bandwidth

    property amplitude:
        def __get__(self): return self.thisptr.amplitude
        def __set__(self, double x): self.thisptr.amplitude = x

    property frequency:
        def __get__(self): return self.thisptr.frequency
        def __set__(self, double x): self.thisptr.frequency = x

    property phase:
        def __get__(self): return self.thisptr.phase
        def __set__(self, double x): self.thisptr.phase = x

    property bandwidth:
        def __get__(self): return self.thisptr.bandwidth
        def __set__(self, double x): self.thisptr.bandwidth = x


cdef class Frame:
    def __cinit__(self, size=2048, create_new=True):
        self._peaks = []
        self._partials = []

        if create_new:
            self.thisptr = new c_Frame(size, True)
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
        p = Peak()
        p.copy(c_p)
        return p

    property peaks:
        def __get__(self):
            self._peaks = [self.peak(i) for i in range(self.thisptr.num_peaks())]
            return self._peaks
        def __set__(self, peaks):
            self.thisptr.clear_peaks()
            self.add_peaks(peaks)
            self._peaks = peaks

    def clear(self):
        self.thisptr.clear()
        self._peaks = []
        self._partials = []

    # partials
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
            peak = Peak()
            peak.copy(c_p)
            return peak
        else:
            self.thisptr.partial(i, p.thisptr)

    property partials:
        def __get__(self):
            self._partials = [self.partial(i) for i in range(self.thisptr.num_partials())]
            return self._partials
        def __set__(self, peaks):
            self.thisptr.clear_partials()
            self.add_partials(peaks)
            self._partials = peaks

    # audio buffers
    property size:
        def __get__(self): return self.thisptr.size()
        def __set__(self, int i): self.thisptr.size(i)

    property synth_size:
        def __get__(self): return self.thisptr.synth_size()
        def __set__(self, int i): self.thisptr.synth_size(i)

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
            shape[0] = <np.npy_intp> self.thisptr.synth_size()
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
            shape[0] = <np.npy_intp> self.thisptr.synth_size()
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.thisptr.synth_residual())
        def __set__(self, np.ndarray[dtype_t, ndim=1] a):
            self.thisptr.synth_residual(<double*> a.data)
