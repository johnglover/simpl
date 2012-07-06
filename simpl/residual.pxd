import numpy as np
cimport numpy as np
np.import_array()
from libcpp.vector cimport vector

from base cimport c_Peak
from base cimport c_Frame
from base cimport string
from base cimport dtype_t
from base import dtype


cdef extern from "../src/simpl/residual.h" namespace "simpl":
    cdef cppclass c_Residual "simpl::Residual":
        c_Residual()
        int frame_size()
        void frame_size(int new_frame_size)
        int next_frame_size()
        int hop_size()
        void hop_size(int new_hop_size)
        int sampling_rate()
        void sampling_rate(int new_sampling_rate)
        void residual_frame(int synth_size, double* synth,
                            int original_size, double* original,
                            int residual_size, double* residual)
        void find_residual(int synth_size, double* synth,
                           int original_size, double* original,
                           int residual_size, double* residual)
        void synth_frame(c_Frame* frame)
        vector[c_Frame*] synth(int synth_size, double* synth,
                               int original_size, double* original)

    cdef cppclass c_SMSResidual "simpl::SMSResidual"(c_Residual):
        c_SMSResidual()
        int num_stochastic_coeffs()
        void num_stochastic_coeffs(int new_num_stochastic_coeffs)
        # int stochastic_type()
