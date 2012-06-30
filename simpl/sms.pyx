import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

from base cimport c_PeakDetection
from base import PeakDetection

np.import_array()


cdef extern from "../src/simpl/simplsms.h" namespace "simpl":
    cdef cppclass c_SMSPeakDetection "simpl::SMSPeakDetection"(c_PeakDetection):
        c_SMSPeakDetection()

        # int sampling_rate()
        # void sampling_rate(int new_sampling_rate)
        # int frame_size()
        # void frame_size(int new_frame_size)
        # int static_frame_size()
        # void static_frame_size(int new_static_frame_size)
        # int next_frame_size()
        # int hop_size()
        # void hop_size(int new_hop_size)
        # int max_peaks()
        # void max_peaks(int new_max_peaks)
        # string window_type()
        # void window_type(string new_window_type)
        # int window_size()
        # void window_size(int new_window_size)
        # double min_peak_separation()
        # void min_peak_separation(double new_min_peak_separation)
        # int num_frames()
        # c_Frame* frame(int frame_number)
        # vector[c_Peak*] find_peaks_in_frame(c_Frame* frame)
        # vector[c_Frame*] find_peaks(int audio_size, double* audio)

    # cdef cppclass c_PartialTracking "simpl::PartialTracking":
    #     c_PartialTracking()
    #     void clear()
    #     int sampling_rate()
    #     void sampling_rate(int new_sampling_rate)
    #     int max_partials()
    #     void max_partials(int new_max_partials)
    #     int min_partial_length()
    #     void min_partial_length(int new_min_partial_length)
    #     int max_gap()
    #     void max_gap(int new_max_gap)
    #     vector[c_Peak*] update_partials(c_Frame* frame)
    #     vector[c_Frame*] find_partials(vector[c_Frame*] frames)

    # cdef cppclass c_Synthesis "simpl::Synthesis":
    #     c_Synthesis()
    #     int frame_size()
    #     void frame_size(int new_frame_size)
    #     int next_frame_size()
    #     int hop_size()
    #     void hop_size(int new_hop_size)
    #     int sampling_rate()
    #     void sampling_rate(int new_sampling_rate)
    #     int max_partials()
    #     void max_partials(int new_max_partials)
    #     void synth_frame(c_Frame* frame)
    #     vector[c_Frame*] synth(vector[c_Frame*] frames)

    # cdef cppclass c_Residual "simpl::Residual":
    #     c_Synthesis()
    #     int frame_size()
    #     void frame_size(int new_frame_size)
    #     int next_frame_size()
    #     int hop_size()
    #     void hop_size(int new_hop_size)
    #     int sampling_rate()
    #     void sampling_rate(int new_sampling_rate)
    #     void residual_frame(int synth_size, double* synth,
    #                         int original_size, double* original,
    #                         int residual_size, double* residual)
    #     void find_residual(int synth_size, double* synth,
    #                        int original_size, double* original,
    #                        int residual_size, double* residual)
    #     void synth_frame(c_Frame* frame)
    #     vector[c_Frame*] synth(vector[c_Frame*] frames)


cdef class SMSPeakDetection(PeakDetection):
    cdef c_SMSPeakDetection* thisptr

    def __cinit__(self): 
        self.thisptr = new c_SMSPeakDetection()
    def __dealloc__(self): del self.thisptr

    # property sampling_rate:
    #     def __get__(self): return self.thisptr.sampling_rate()
    #     def __set__(self, int i): self.thisptr.sampling_rate(i)

    # property frame_size:
    #     def __get__(self): return self.thisptr.frame_size()
    #     def __set__(self, int i): self.thisptr.frame_size(i)

    # property static_frame_size:
    #     def __get__(self): return self.thisptr.static_frame_size()
    #     def __set__(self, int i): self.thisptr.static_frame_size(i)

    # def next_frame_size(self):
    #     return self.thisptr.next_frame_size()

    # property hop_size:
    #     def __get__(self): return self.thisptr.hop_size()
    #     def __set__(self, int i): self.thisptr.hop_size(i)

    # property max_peaks:
    #     def __get__(self): return self.thisptr.max_peaks()
    #     def __set__(self, int i): self.thisptr.max_peaks(i)

    # property window_type:
    #     def __get__(self): return self.thisptr.window_type().c_str()
    #     def __set__(self, char* s): self.thisptr.window_type(string(s))

    # property window_size:
    #     def __get__(self): return self.thisptr.window_size()
    #     def __set__(self, int i): self.thisptr.window_size(i)

    # property min_peak_separation:
    #     def __get__(self): return self.thisptr.min_peak_separation()
    #     def __set__(self, double d): self.thisptr.min_peak_separation(d)

    # def frame(self, int i):
    #     cdef c_Frame* c_f = self.thisptr.frame(i)
    #     f = Frame(None, False)
    #     f.set_frame(c_f)
    #     return f

    # property frames:
    #     def __get__(self):
    #         return [self.frame(i) for i in range(self.thisptr.num_frames())]
    #     def __set__(self, f):
    #         raise Exception("NotImplemented")

    # def find_peaks_in_frame(self, Frame frame not None):
    #     peaks = []
    #     cdef vector[c_Peak*] c_peaks = self.thisptr.find_peaks_in_frame(frame.thisptr)
    #     for i in range(c_peaks.size()):
    #         peak = Peak(False)
    #         peak.set_peak(c_peaks[i])
    #         peaks.append(peak)
    #     return peaks

    # def find_peaks(self, np.ndarray[dtype_t, ndim=1] audio):
    #     frames = []
    #     cdef vector[c_Frame*] output_frames = self.thisptr.find_peaks(len(audio), <double*> audio.data)
    #     for i in range(output_frames.size()):
    #         f = Frame(output_frames[i].size(), False)
    #         f.set_frame(output_frames[i])
    #         frames.append(f)
    #     return frames


# cdef class PartialTracking:
#     cdef c_PartialTracking* thisptr

#     def __cinit__(self): self.thisptr = new c_PartialTracking()
#     def __dealloc__(self): del self.thisptr

#     def clear(self):
#         self.thisptr.clear()

#     property sampling_rate:
#         def __get__(self): return self.thisptr.sampling_rate()
#         def __set__(self, int i): self.thisptr.sampling_rate(i)

#     property max_partials:
#         def __get__(self): return self.thisptr.max_partials()
#         def __set__(self, int i): self.thisptr.max_partials(i)

#     property min_partial_length:
#         def __get__(self): return self.thisptr.min_partial_length()
#         def __set__(self, int i): self.thisptr.min_partial_length(i)

#     property max_gap:
#         def __get__(self): return self.thisptr.max_gap()
#         def __set__(self, int i): self.thisptr.max_gap(i)

#     def update_partials(self, Frame frame not None):
#         peaks = []
#         cdef vector[c_Peak*] c_peaks = self.thisptr.update_partials(frame.thisptr)
#         for i in range(c_peaks.size()):
#             peak = Peak(False)
#             peak.set_peak(c_peaks[i])
#             peaks.append(peak)
#         return peaks

#     def find_partials(self, frames):
#         partial_frames = []
#         cdef vector[c_Frame*] c_frames
#         for frame in frames:
#             c_frames.push_back((<Frame>frame).thisptr)
#         cdef vector[c_Frame*] output_frames = self.thisptr.find_partials(c_frames)
#         for i in range(output_frames.size()):
#             f = Frame(output_frames[i].size(), False)
#             f.set_frame(output_frames[i])
#             partial_frames.append(f)
#         return partial_frames


# cdef class Synthesis:
#     cdef c_Synthesis* thisptr

#     def __cinit__(self): self.thisptr = new c_Synthesis()
#     def __dealloc__(self): del self.thisptr

#     property sampling_rate:
#         def __get__(self): return self.thisptr.sampling_rate()
#         def __set__(self, int i): self.thisptr.sampling_rate(i)

#     property frame_size:
#         def __get__(self): return self.thisptr.frame_size()
#         def __set__(self, int i): self.thisptr.frame_size(i)

#     property hop_size:
#         def __get__(self): return self.thisptr.hop_size()
#         def __set__(self, int i): self.thisptr.hop_size(i)

#     property max_partials:
#         def __get__(self): return self.thisptr.max_partials()
#         def __set__(self, int i): self.thisptr.max_partials(i)

#     def synth_frame(self, Frame frame not None):
#         self.thisptr.synth_frame(frame.thisptr)
#         return frame.audio

#     def synth(self, frames):
#         cdef vector[c_Frame*] c_frames
#         for frame in frames:
#             c_frames.push_back((<Frame>frame).thisptr)
#         cdef vector[c_Frame*] output_frames = self.thisptr.synth(c_frames)
#         cdef np.ndarray[dtype_t, ndim=1] output = np.zeros(
#             output_frames.size() * self.thisptr.hop_size()
#         )
#         cdef np.npy_intp shape[1]
#         shape[0] = <np.npy_intp> self.thisptr.hop_size()
#         for i in range(output_frames.size()):
#             frame_audio = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, output_frames[i].synth())
#             output[i * self.thisptr.hop_size():(i + 1) * self.thisptr.hop_size()] = frame_audio
#         return output


# cdef class Residual:
#     cdef c_Residual* thisptr

#     def __cinit__(self): self.thisptr = new c_Residual()
#     def __dealloc__(self): del self.thisptr

#     property sampling_rate:
#         def __get__(self): return self.thisptr.sampling_rate()
#         def __set__(self, int i): self.thisptr.sampling_rate(i)

#     property frame_size:
#         def __get__(self): return self.thisptr.frame_size()
#         def __set__(self, int i): self.thisptr.frame_size(i)

#     property hop_size:
#         def __get__(self): return self.thisptr.hop_size()
#         def __set__(self, int i): self.thisptr.hop_size(i)

#     def residual_frame(self, np.ndarray[dtype_t, ndim=1] synth,
#                        np.ndarray[dtype_t, ndim=1] original):
#         cdef np.ndarray[dtype_t, ndim=1] residual = np.zeros(len(synth))
#         self.thisptr.residual_frame(len(synth), <double*> synth.data,
#                                     len(original), <double*> original.data,
#                                     len(residual), <double*> residual.data)
#         return residual

#     def find_residual(self, np.ndarray[dtype_t, ndim=1] synth,
#                       np.ndarray[dtype_t, ndim=1] original):
#         cdef np.ndarray[dtype_t, ndim=1] residual = np.zeros(len(synth))
#         self.thisptr.find_residual(len(synth), <double*> synth.data,
#                                    len(original), <double*> original.data,
#                                    len(residual), <double*> residual.data)
#         return residual

#     def synth_frame(self, Frame frame not None):
#         self.thisptr.synth_frame(frame.thisptr)
#         return frame.audio

#     def synth(self, frames):
#         cdef vector[c_Frame*] c_frames
#         for frame in frames:
#             c_frames.push_back((<Frame>frame).thisptr)
#         cdef vector[c_Frame*] output_frames = self.thisptr.synth(c_frames)
#         cdef np.ndarray[dtype_t, ndim=1] output = np.zeros(
#             output_frames.size() * self.thisptr.hop_size()
#         )
#         cdef np.npy_intp shape[1]
#         shape[0] = <np.npy_intp> self.thisptr.hop_size()
#         for i in range(output_frames.size()):
#             frame_audio = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, output_frames[i].synth())
#             output[i * self.thisptr.hop_size():(i + 1) * self.thisptr.hop_size()] = frame_audio
#         return output
