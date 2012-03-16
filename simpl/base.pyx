import numpy as np
cimport numpy as np
np.import_array()

dtype = np.float64
ctypedef np.double_t dtype_t

cdef extern from "../src/simpl/base.h" namespace "simpl": 
    cdef cppclass c_Frame "simpl::Frame":
        c_Frame()
        c_Frame(int frame_size)

        # peaks
        # int num_peaks()
        # int max_peaks()
        # void max_peaks(int new_max_peaks)
        # void add_peak(Peak peak)
        # void add_peaks(Peaks* peaks)
        # Peak peak(int peak_number)
        # void clear_peaks()
        # Peaks::iterator peaks_begin()
        # Peaks::iterator peaks_end()

        # partials
        # int num_partials()
        # int max_partials()
        # void max_partials(int new_max_partials)
        # void add_partial(Partial partial)
        # Partials::iterator partials()

        # audio buffers
        int size()
        void size(int new_size)
        void audio(double* new_audio)
        double* audio()
        void synth(double* new_synth)
        double* synth()
        void residual(double* new_residual)
        double* residual()
        void synth_residual(double* new_synth_residual)
        double* synth_residual()

cdef class Frame:
    cdef c_Frame *thisptr

    def __cinit__(self, size=None):
        if size:
            self.thisptr = new c_Frame(size)
        else:
            self.thisptr = new c_Frame()

    def __dealloc__(self):
        del self.thisptr

    property size:
        def __get__(self): return self.thisptr.size()
        def __set__(self, int n): self.thisptr.size(n)

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

