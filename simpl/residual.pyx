import numpy as np
cimport numpy as np
np.import_array()
from libcpp.vector cimport vector

from base cimport Peak
from base cimport Frame
from base cimport c_Peak
from base cimport c_Frame


cdef class Residual:
    cdef c_Residual* thisptr

    def __cinit__(self):
        self.thisptr = new c_Residual()

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
        def __set__(self, int i):
            print 'setting hop size...'
            self.thisptr.hop_size(i)

    def residual_frame(self, Frame frame not None):
        self.thisptr.residual_frame(frame.thisptr)
        return frame.residual

    def find_residual(self, np.ndarray[dtype_t, ndim=1] synth,
                      np.ndarray[dtype_t, ndim=1] original):
        cdef np.ndarray[dtype_t, ndim=1] residual = np.zeros(len(synth))
        self.thisptr.find_residual(len(synth), <double*> synth.data,
                                   len(original), <double*> original.data,
                                   len(residual), <double*> residual.data)
        return residual

    def synth_frame(self, Frame frame not None):
        self.thisptr.synth_frame(frame.thisptr)
        return frame.audio

    def synth(self, np.ndarray[dtype_t, ndim=1] original):
        cdef int hop = self.thisptr.hop_size()
        cdef vector[c_Frame*] output_frames = self.thisptr.synth(
            len(original), <double*> original.data
        )

        cdef np.ndarray[dtype_t, ndim=1] output = np.zeros(output_frames.size() * hop)
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> hop

        for i in range(output_frames.size()):
            frame_audio = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, output_frames[i].synth_residual())
            output[i * hop:(i + 1) * hop] = frame_audio

        return output


cdef class SMSResidual(Residual):
    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new c_SMSResidual()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <c_Residual*>0

    property hop_size:
        def __get__(self): return self.thisptr.hop_size()
        def __set__(self, int i): self.thisptr.hop_size(i)

    property num_stochastic_coeffs:
        def __get__(self): return (<c_SMSResidual*>self.thisptr).num_stochastic_coeffs()
        def __set__(self, int i): (<c_SMSResidual*>self.thisptr).num_stochastic_coeffs(i)
