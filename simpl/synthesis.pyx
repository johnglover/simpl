import numpy as np
cimport numpy as np
np.import_array()
from libcpp.vector cimport vector

from base cimport Peak
from base cimport Frame
from base cimport c_Peak
from base cimport c_Frame


cdef class Synthesis:
    cdef c_Synthesis* thisptr

    def __cinit__(self):
        self.thisptr = new c_Synthesis()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr

    property sampling_rate:
        def __get__(self): return self.thisptr.sampling_rate()
        def __set__(self, int i): self.thisptr.sampling_rate(i)

    property frame_size:
        def __get__(self): return self.thisptr.frame_size()
        def __set__(self, int i): self.thisptr.frame_size(i)

    property hop_size:
        def __get__(self): return self.thisptr.hop_size()
        def __set__(self, int i): self.thisptr.hop_size(i)

    property max_partials:
        def __get__(self): return self.thisptr.max_partials()
        def __set__(self, int i): self.thisptr.max_partials(i)

    def synth_frame(self, Frame frame not None):
        self.thisptr.synth_frame(frame.thisptr)
        return frame.synth

    def synth(self, frames):
        cdef int size = self.thisptr.hop_size()
        cdef np.ndarray[dtype_t, ndim=1] output = np.zeros(len(frames) * size)
        for i in range(len(frames)):
            frames[i].synth = np.zeros(size)
            frames[i].synth_size = size
            self.synth_frame(frames[i])
            output[i * size:(i + 1) * size] = frames[i].synth
        return output


cdef class SMSSynthesis(Synthesis):
    SMS_DET_IFFT = 0
    SMS_DET_SIN = 1

    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new c_SMSSynthesis()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <c_Synthesis*>0

    property num_stochastic_coeffs:
        def __get__(self): return (<c_SMSSynthesis*>self.thisptr).num_stochastic_coeffs()
        def __set__(self, int i): raise Exception("NotImplemented")

    property stochastic_type:
        def __get__(self): return (<c_SMSSynthesis*>self.thisptr).stochastic_type()
        def __set__(self, int i): raise Exception("NotImplemented")

    property det_synthesis_type:
        def __get__(self): return (<c_SMSSynthesis*>self.thisptr).det_synthesis_type()
        def __set__(self, int i): (<c_SMSSynthesis*>self.thisptr).det_synthesis_type(i)


cdef class SndObjSynthesis(Synthesis):
    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new c_SndObjSynthesis()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <c_Synthesis*>0


cdef class LorisSynthesis(Synthesis):
    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new c_LorisSynthesis()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <c_Synthesis*>0

    property bandwidth:
        def __get__(self): return (<c_LorisSynthesis*>self.thisptr).bandwidth()
        def __set__(self, double d): (<c_LorisSynthesis*>self.thisptr).bandwidth(d)
