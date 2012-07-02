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

    def __cinit__(self):
        self.thisptr = new c_PeakDetection()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr

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

    property frames:
        def __get__(self):
            return [self.frame(i) for i in range(self.thisptr.num_frames())]
        def __set__(self, new_frames):
            cdef vector[c_Frame*] c_frames
            for f in new_frames:
                c_frames.push_back((<Frame>f).thisptr)
            self.thisptr.frames(c_frames)

    def find_peaks_in_frame(self, Frame frame not None):
        peaks = []
        cdef vector[c_Peak*] c_peaks = self.thisptr.find_peaks_in_frame(frame.thisptr)
        for i in range(c_peaks.size()):
            peak = Peak(False)
            peak.set_peak(c_peaks[i])
            peaks.append(peak)
        return peaks

    def find_peaks(self, np.ndarray[dtype_t, ndim=1] audio):
        frames = []
        cdef vector[c_Frame*] output_frames = self.thisptr.find_peaks(len(audio), <double*> audio.data)
        for i in range(output_frames.size()):
            f = Frame(output_frames[i].size(), False)
            f.set_frame(output_frames[i])
            frames.append(f)
        return frames


cdef class SMSPeakDetection(PeakDetection):
    def __cinit__(self):
        if self.thisptr:
            del self.thisptr
        self.thisptr = new c_SMSPeakDetection()

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr
            self.thisptr = <c_PeakDetection*>0
