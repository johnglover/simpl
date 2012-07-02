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
        return frame.audio

    def synth(self, frames):
        cdef vector[c_Frame*] c_frames
        for frame in frames:
            c_frames.push_back((<Frame>frame).thisptr)
        cdef vector[c_Frame*] output_frames = self.thisptr.synth(c_frames)
        cdef np.ndarray[dtype_t, ndim=1] output = np.zeros(
            output_frames.size() * self.thisptr.hop_size()
        )
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.thisptr.hop_size()
        for i in range(output_frames.size()):
            frame_audio = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, output_frames[i].synth())
            output[i * self.thisptr.hop_size():(i + 1) * self.thisptr.hop_size()] = frame_audio
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
