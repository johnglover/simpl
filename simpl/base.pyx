import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

np.import_array()

dtype = np.float64
ctypedef np.double_t dtype_t


cdef extern from "<string>" namespace "std":
    cdef cppclass string:
        string()
        string(char *)
        char * c_str()


cdef extern from "../src/simpl/base.h" namespace "simpl": 
    cdef cppclass c_Peak "simpl::Peak":
        c_Peak()
        double amplitude
        double frequency
        double phase

    cdef cppclass c_Frame "simpl::Frame":
        c_Frame()
        c_Frame(int frame_size)

        # peaks
        int num_peaks()
        int max_peaks()
        void max_peaks(int new_max_peaks)
        void add_peak(c_Peak* peak)
        # void add_peaks(Peaks* peaks)
        c_Peak* peak(int peak_number)
        void clear_peaks()
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

    cdef cppclass c_PeakDetection "simpl::PeakDetection":
        c_PeakDetection()

        int sampling_rate()
        void sampling_rate(int new_sampling_rate)
        int frame_size()
        void frame_size(int new_frame_size)
        int static_frame_size()
        void static_frame_size(int new_static_frame_size)
        int next_frame_size()
        int hop_size()
        void hop_size(int new_hop_size)
        int max_peaks()
        void max_peaks(int new_max_peaks)
        string window_type()
        void window_type(string new_window_type)
        int window_size()
        void window_size(int new_window_size)
        double min_peak_separation()
        void min_peak_separation(double new_min_peak_separation)
        # Frames* frames()
        # virtual Peaks* find_peaks_in_frame(const Frame& frame)
        # virtual Frames* find_peaks(number* audio)


cdef class Peak:
    cdef c_Peak* thisptr
    cdef int peak_created

    def __cinit__(self, create_new=True): 
        if create_new:
            self.thisptr = new c_Peak()
            self.peak_created = True
        else:
            self.peak_created = False

    def __dealloc__(self): 
        if self.peak_created:
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
    cdef c_Frame* thisptr

    def __cinit__(self, size=None):
        if size:
            self.thisptr = new c_Frame(size)
        else:
            self.thisptr = new c_Frame()

    def __dealloc__(self): del self.thisptr

    # peaks
    property num_peaks:
        def __get__(self): return self.thisptr.num_peaks()

    property max_peaks:
        def __get__(self): return self.thisptr.max_peaks()
        def __set__(self, int i): self.thisptr.max_peaks(i)

    def add_peak(self, Peak p not None):
        self.thisptr.add_peak(p.thisptr)

    def peak(self, int i):
        cdef c_Peak* c_p = self.thisptr.peak(i)
        p = Peak(False)
        p.set_peak(c_p)
        return p

    property peaks:
        def __get__(self):
            return [self.peak(i) for i in range(self.thisptr.num_peaks())]

    def clear_peaks(self):
        self.thisptr.clear_peaks()

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


cdef class PeakDetection:
    cdef c_PeakDetection* thisptr

    def __cinit__(self): self.thisptr = new c_PeakDetection()
    def __dealloc__(self): del self.thisptr

    property sampling_rate:
        def __get__(self): return self.thisptr.sampling_rate()
        def __set__(self, int i): self.thisptr.sampling_rate(i)

    property frame_size:
        def __get__(self): return self.thisptr.frame_size()
        def __set__(self, int i): self.thisptr.frame_size(i)

    property static_frame_size:
        def __get__(self): return self.thisptr.static_frame_size()
        def __set__(self, int i): self.thisptr.static_frame_size(i)

    def next_frame_size(self):
        return self.thisptr.next_frame_size()

    property hop_size:
        def __get__(self): return self.thisptr.hop_size()
        def __set__(self, int i): self.thisptr.hop_size(i)

    property max_peaks:
        def __get__(self): return self.thisptr.max_peaks()
        def __set__(self, int i): self.thisptr.max_peaks(i)

    property window_type:
        def __get__(self): return self.thisptr.window_type().c_str()
        def __set__(self, char* s): self.thisptr.window_type(string(s))

    property window_size:
        def __get__(self): return self.thisptr.window_size()
        def __set__(self, int i): self.thisptr.window_size(i)

    property min_peak_separation:
        def __get__(self): return self.thisptr.min_peak_separation()
        def __set__(self, double d): self.thisptr.min_peak_separation(d)
