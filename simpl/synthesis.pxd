import numpy as np
cimport numpy as np
np.import_array()
from libcpp.vector cimport vector

from base cimport c_Peak
from base cimport c_Frame
from base cimport string
from base cimport dtype_t
from base import dtype


cdef extern from "../src/simpl/synthesis.h" namespace "simpl":
    cdef cppclass c_Synthesis "simpl::Synthesis":
        c_Synthesis()
        int frame_size()
        void frame_size(int new_frame_size)
        int next_frame_size()
        int hop_size()
        void hop_size(int new_hop_size)
        int sampling_rate()
        void sampling_rate(int new_sampling_rate)
        int max_partials()
        void max_partials(int new_max_partials)
        void synth_frame(c_Frame* frame)
        vector[c_Frame*] synth(vector[c_Frame*] frames)
