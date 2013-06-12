import numpy as np
cimport numpy as np
np.import_array()
from libcpp.vector cimport vector
from libcpp cimport bool

from base cimport Peak
from base cimport Frame
from base cimport c_Peak
from base cimport c_Frame


cdef class PeakDetection:
    cdef c_PeakDetection* thisptr
    cdef public list frames

    def __cinit__(self):
        self.thisptr = new c_PeakDetection()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr

    def __init__(self):
        self.frames = []

    property sampling_rate:
        def __get__(self): return self.thisptr.sampling_rate()
        def __set__(self, int i): self.thisptr.sampling_rate(i)

    property frame_size:
        def __get__(self): return self.thisptr.frame_size()
        def __set__(self, int i): self.thisptr.frame_size(i)

    property static_frame_size:
        def __get__(self): return self.thisptr.static_frame_size()
        def __set__(self, bool b): self.thisptr.static_frame_size(b)

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

    def frame(self, int i):
        cdef c_Frame* c_f = self.thisptr.frame(i)
        f = Frame(None, False)
        f.set_frame(c_f)
        return f

    def find_peaks_in_frame(self, Frame frame not None):
        self.thisptr.find_peaks_in_frame(frame.thisptr)
        return frame.peaks

    def find_peaks(self, np.ndarray[dtype_t, ndim=1] audio):
        self.frames = []

        cdef int pos = 0
        while pos <= len(audio) - self.hop_size:
            if not self.static_frame_size:
                self.frame_size = self.next_frame_size()

            frame = Frame(self.frame_size)

            if pos < len(audio) - self.frame_size:
                frame.audio = audio[pos:pos + self.frame_size]
            else:
                frame.audio = np.hstack((
                    audio[pos:len(audio)],
                    np.zeros(self.frame_size - (len(audio) - pos))
                ))

            frame.max_peaks = self.max_peaks
            self.find_peaks_in_frame(frame)
            self.frames.append(frame)
            pos += self.hop_size

        return self.frames


cdef class MQPeakDetection(PeakDetection):
    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new c_MQPeakDetection()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <c_PeakDetection*>0


cdef class SMSPeakDetection(PeakDetection):
    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new c_SMSPeakDetection()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <c_PeakDetection*>0

    def find_peaks(self, np.ndarray[dtype_t, ndim=1] audio):
        self.frames = []
        cdef vector[c_Frame*] output_frames = \
            self.thisptr.find_peaks(len(audio), <double*> audio.data)
        for i in range(output_frames.size()):
            f = Frame(output_frames[i].size(), False)
            f.set_frame(output_frames[i])
            self.frames.append(f)
        return self.frames


cdef class SndObjPeakDetection(PeakDetection):
    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new c_SndObjPeakDetection()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <c_PeakDetection*>0


cdef class LorisPeakDetection(PeakDetection):
    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new c_LorisPeakDetection()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <c_PeakDetection*>0
