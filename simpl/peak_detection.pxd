import numpy as np
cimport numpy as np
np.import_array()
from libcpp.vector cimport vector
from libcpp cimport bool

from base cimport c_Peak
from base cimport c_Frame
from base cimport string
from base cimport dtype_t
from base import dtype


cdef extern from "../src/simpl/peak_detection.h" namespace "simpl":
    cdef cppclass c_PeakDetection "simpl::PeakDetection":
        c_PeakDetection()
        int sampling_rate()
        void sampling_rate(int new_sampling_rate)
        int frame_size()
        void frame_size(int new_frame_size)
        int static_frame_size()
        void static_frame_size(bool new_static_frame_size)
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
        int num_frames()
        c_Frame* frame(int frame_number)
        void frames(vector[c_Frame*] new_frames)
        vector[c_Peak*] find_peaks_in_frame(c_Frame* frame)
        vector[c_Frame*] find_peaks(int audio_size, double* audio)

    cdef cppclass c_MQPeakDetection "simpl::MQPeakDetection"(c_PeakDetection):
        c_MQPeakDetection()
        void hop_size(int new_hop_size)
        void max_peaks(int new_max_peaks)
        vector[c_Peak*] find_peaks_in_frame(c_Frame* frame)

    cdef cppclass c_SMSPeakDetection "simpl::SMSPeakDetection"(c_PeakDetection):
        c_SMSPeakDetection()
        void hop_size(int new_hop_size)
        void max_peaks(int new_max_peaks)
        vector[c_Peak*] find_peaks_in_frame(c_Frame* frame)
        vector[c_Frame*] find_peaks(int audio_size, double* audio)

    cdef cppclass c_SndObjPeakDetection "simpl::SndObjPeakDetection"(c_PeakDetection):
        c_SndObjPeakDetection()
        void hop_size(int new_hop_size)
        void max_peaks(int new_max_peaks)
        vector[c_Peak*] find_peaks_in_frame(c_Frame* frame)

    cdef cppclass c_LorisPeakDetection "simpl::LorisPeakDetection"(c_PeakDetection):
        c_LorisPeakDetection()
        void hop_size(int new_hop_size)
        void max_peaks(int new_max_peaks)
        vector[c_Peak*] find_peaks_in_frame(c_Frame* frame)
