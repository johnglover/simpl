import numpy as np
cimport numpy as np
np.import_array()
from libcpp.vector cimport vector

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
        vector[c_Peak*] update_partials(c_Frame* frame)
        vector[c_Frame*] find_partials(vector[c_Frame*] frames)

    cdef cppclass c_SMSPartialTracking "simpl::SMSPartialTracking"(c_PartialTracking):
        c_SMSPartialTracking()

    cdef cppclass c_SndObjPartialTracking "simpl::SndObjPartialTracking"(c_PartialTracking):
        c_SndObjPartialTracking()

    cdef cppclass c_LorisPartialTracking "simpl::LorisPartialTracking"(c_PartialTracking):
        c_LorisPartialTracking()
