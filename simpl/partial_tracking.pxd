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


cdef extern from "../src/simpl/partial_tracking.h" namespace "simpl":
    cdef cppclass c_PartialTracking "simpl::PartialTracking":
        c_PartialTracking()
        void clear()
        int sampling_rate()
        void sampling_rate(int new_sampling_rate)
        int max_partials()
        void max_partials(int new_max_partials)
        int min_partial_length()
        void min_partial_length(int new_min_partial_length)
        int max_gap()
        void max_gap(int new_max_gap)
        void update_partials(c_Frame* frame)
        vector[c_Frame*] find_partials(vector[c_Frame*] frames)

    cdef cppclass c_MQPartialTracking "simpl::MQPartialTracking"(c_PartialTracking):
        c_MQPartialTracking()

    cdef cppclass c_SMSPartialTracking "simpl::SMSPartialTracking"(c_PartialTracking):
        c_SMSPartialTracking()
        bool realtime()
        void realtime(bool is_realtime)
        bool harmonic()
        void harmonic(bool is_harmonic)
        int max_frame_delay()
        void max_frame_delay(int new_max_frame_delay)
        int analysis_delay()
        void analysis_delay(int new_analysis_delay)
        int min_good_frames()
        void min_good_frames(int new_min_good_frames)
        bool clean_tracks()
        void clean_tracks(bool new_clean_tracks)

    cdef cppclass c_SndObjPartialTracking "simpl::SndObjPartialTracking"(c_PartialTracking):
        c_SndObjPartialTracking()

    cdef cppclass c_LorisPartialTracking "simpl::LorisPartialTracking"(c_PartialTracking):
        c_LorisPartialTracking()
