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

    cdef cppclass c_SMSSynthesis "simpl::SMSSynthesis"(c_Synthesis):
        c_SMSSynthesis()
        int num_stochastic_coeffs()
        int stochastic_type()
        int det_synthesis_type()
        void det_synthesis_type(int new_det_synthesis_type)

    cdef cppclass c_SndObjSynthesis "simpl::SndObjSynthesis"(c_Synthesis):
        c_SndObjSynthesis()

    cdef cppclass c_LorisSynthesis "simpl::LorisSynthesis"(c_Synthesis):
        c_LorisSynthesis()
        double bandwidth()
        void bandwidth(double new_bandwidth)
