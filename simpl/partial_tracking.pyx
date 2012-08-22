import numpy as np
cimport numpy as np
np.import_array()
from libcpp.vector cimport vector

from base cimport Peak
from base cimport Frame
from base cimport c_Peak
from base cimport c_Frame


cdef class PartialTracking:
    cdef c_PartialTracking* thisptr

    def __cinit__(self):
        self.thisptr = new c_PartialTracking()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr

    def clear(self):
        self.thisptr.clear()

    property sampling_rate:
        def __get__(self): return self.thisptr.sampling_rate()
        def __set__(self, int i): self.thisptr.sampling_rate(i)

    property max_partials:
        def __get__(self): return self.thisptr.max_partials()
        def __set__(self, int i): self.thisptr.max_partials(i)

    property min_partial_length:
        def __get__(self): return self.thisptr.min_partial_length()
        def __set__(self, int i): self.thisptr.min_partial_length(i)

    property max_gap:
        def __get__(self): return self.thisptr.max_gap()
        def __set__(self, int i): self.thisptr.max_gap(i)

    def update_partials(self, Frame frame not None):
        peaks = []
        cdef vector[c_Peak*] c_peaks = self.thisptr.update_partials(frame.thisptr)
        for i in range(c_peaks.size()):
            peak = Peak(False)
            peak.set_peak(c_peaks[i])
            peaks.append(peak)
        frame.partials = peaks
        return peaks

    def find_partials(self, frames):
        partial_frames = []
        for frame in frames:
            if frame.max_partials != self.thisptr.max_partials():
                frame.max_partials = self.thisptr.max_partials()
            self.update_partials(frame)
            partial_frames.append(frame)
        return partial_frames


cdef class SMSPartialTracking(PartialTracking):
    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new c_SMSPartialTracking()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <c_PartialTracking*>0


cdef class SndObjPartialTracking(PartialTracking):
    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new c_SndObjPartialTracking()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <c_PartialTracking*>0


cdef class LorisPartialTracking(PartialTracking):
    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new c_LorisPartialTracking()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <c_PartialTracking*>0
