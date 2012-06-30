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

    def __cinit__(self): self.thisptr = new c_PartialTracking()
    def __dealloc__(self): del self.thisptr

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
        return peaks

    def find_partials(self, frames):
        partial_frames = []
        cdef vector[c_Frame*] c_frames
        for frame in frames:
            c_frames.push_back((<Frame>frame).thisptr)
        cdef vector[c_Frame*] output_frames = self.thisptr.find_partials(c_frames)
        for i in range(output_frames.size()):
            f = Frame(output_frames[i].size(), False)
            f.set_frame(output_frames[i])
            partial_frames.append(f)
        return partial_frames
