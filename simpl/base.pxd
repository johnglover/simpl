import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool

np.import_array()

dtype = np.float64
ctypedef np.double_t dtype_t


cdef class Peak:
    cdef c_Peak* thisptr
    cdef int created
    cdef set_peak(self, c_Peak* p)


cdef class Frame:
    cdef c_Frame* thisptr
    cdef int created
    cdef set_frame(self, c_Frame* f)
    cdef list _peaks


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
        c_Frame(int frame_size, bool alloc_memory)

        # peaks
        int num_peaks()
        void num_peaks(int new_num_peaks)
        int max_peaks()
        void max_peaks(int new_max_peaks)
        c_Peak* peak(int peak_number)
        void clear()

        # partials
        int num_partials()
        void num_partials(int new_num_partials)
        int max_partials()
        void max_partials(int new_max_partials)
        void add_partial(c_Peak* peak)
        c_Peak* partial(int partial_number)
        void partial(int partial_number, c_Peak* peak)

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
