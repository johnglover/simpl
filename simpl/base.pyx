import numpy as np
cimport numpy as np
from libcpp.vector cimport vector

np.import_array()

dtype = np.float64
ctypedef np.double_t dtype_t


cdef extern from "<string>" namespace "std":
    cdef cppclass string:
        string()
        string(char *)
        char * c_str()


cdef extern from "../src/simpl/base.h" namespace "simpl":
    cdef cppclass c_Peak "simpl::Peak":
        c_Peak()
        double amplitude
        double frequency
        double phase

    cdef cppclass c_Frame "simpl::Frame":
        c_Frame()
        c_Frame(int frame_size)

        # peaks
        int num_peaks()
        int max_peaks()
        void max_peaks(int new_max_peaks)
        void add_peak(c_Peak* peak)
        c_Peak* peak(int peak_number)
        void clear()

        # partials
        int num_partials()
        int max_partials()
        void max_partials(int new_max_partials)
        c_Peak* partial(int partial_number)
        void partial(int partial_number, c_Peak* peak)

        # audio buffers
        int size()
        void size(int new_size)
        void audio(double* new_audio)
        double* audio()
        void synth(double* new_synth)
        double* synth()
        void residual(double* new_residual)
        double* residual()
        void synth_residual(double* new_synth_residual)
        double* synth_residual()

    cdef cppclass c_PeakDetection "simpl::PeakDetection":
        c_PeakDetection()

        int sampling_rate()
        void sampling_rate(int new_sampling_rate)
        int frame_size()
        void frame_size(int new_frame_size)
        int static_frame_size()
        void static_frame_size(int new_static_frame_size)
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
        vector[c_Peak*] find_peaks_in_frame(c_Frame* frame)
        vector[c_Frame*] find_peaks(int audio_size, double* audio)

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

    cdef cppclass c_Residual "simpl::Residual":
        c_Synthesis()
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
        vector[c_Frame*] synth(vector[c_Frame*] frames)


cdef class Peak:
    cdef c_Peak* thisptr
    cdef int created

    def __cinit__(self, create_new=True):
        if create_new:
            self.thisptr = new c_Peak()
            self.created = True
        else:
            self.created = False

    def __dealloc__(self):
        if self.created:
            del self.thisptr

    cdef set_peak(self, c_Peak* p):
        self.thisptr = p

    property amplitude:
        def __get__(self): return self.thisptr.amplitude
        def __set__(self, double x): self.thisptr.amplitude = x

    property frequency:
        def __get__(self): return self.thisptr.frequency
        def __set__(self, double x): self.thisptr.frequency = x

    property phase:
        def __get__(self): return self.thisptr.phase
        def __set__(self, double x): self.thisptr.phase = x


cdef class Frame:
    cdef c_Frame* thisptr
    cdef int created

    def __cinit__(self, size=None, create_new=True):
        if create_new:
            if size:
                self.thisptr = new c_Frame(size)
            else:
                self.thisptr = new c_Frame()
            self.created = True
        else:
            self.created = False

    def __dealloc__(self):
        if self.created:
            del self.thisptr

    cdef set_frame(self, c_Frame* f):
        self.thisptr = f

    # peaks
    property num_peaks:
        def __get__(self): return self.thisptr.num_peaks()

    property max_peaks:
        def __get__(self): return self.thisptr.max_peaks()
        def __set__(self, int i): self.thisptr.max_peaks(i)

    def add_peak(self, Peak p not None):
        self.thisptr.add_peak(p.thisptr)

    def add_peaks(self, peaks not None):
        for p in peaks:
            self.add_peak(p)

    def peak(self, int i):
        cdef c_Peak* c_p = self.thisptr.peak(i)
        p = Peak(False)
        p.set_peak(c_p)
        return p

    property peaks:
        def __get__(self):
            return [self.peak(i) for i in range(self.thisptr.num_peaks())]
        def __set__(self, peaks):
            self.add_peaks(peaks)

    def clear(self):
        self.thisptr.clear()

    # partials
    property num_partials:
        def __get__(self): return self.thisptr.num_partials()
        def __set__(self, int i): raise Exception("Invalid Operation")

    property max_partials:
        def __get__(self): return self.thisptr.max_partials()
        def __set__(self, int i): self.thisptr.max_partials(i)

    def partial(self, int i, Peak p=None):
        cdef c_Peak* c_p
        if not p:
            c_p = self.thisptr.partial(i)
            peak = Peak(False)
            peak.set_peak(c_p)
            return peak
        else:
            self.thisptr.partial(i, p.thisptr)

    property partials:
        def __get__(self):
            return [self.partial(i) for i in range(self.thisptr.num_partials())]
        def __set__(self, peaks):
            raise Exception("NotImplemented")

    # audio buffers
    property size:
        def __get__(self): return self.thisptr.size()
        def __set__(self, int i): self.thisptr.size(i)

    property audio:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.thisptr.size()
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.thisptr.audio())
        def __set__(self, np.ndarray[dtype_t, ndim=1] a):
            self.thisptr.audio(<double*> a.data)

    property synth:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.thisptr.size()
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.thisptr.synth())
        def __set__(self, np.ndarray[dtype_t, ndim=1] a):
            self.thisptr.synth(<double*> a.data)

    property residual:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.thisptr.size()
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.thisptr.residual())
        def __set__(self, np.ndarray[dtype_t, ndim=1] a):
            self.thisptr.residual(<double*> a.data)

    property synth_residual:
        def __get__(self):
            cdef np.npy_intp shape[1]
            shape[0] = <np.npy_intp> self.thisptr.size()
            return np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, self.thisptr.synth_residual())
        def __set__(self, np.ndarray[dtype_t, ndim=1] a):
            self.thisptr.synth_residual(<double*> a.data)


cdef class PeakDetection:
    cdef c_PeakDetection* thisptr

    def __cinit__(self): self.thisptr = new c_PeakDetection()
    def __dealloc__(self): del self.thisptr

    property sampling_rate:
        def __get__(self): return self.thisptr.sampling_rate()
        def __set__(self, int i): self.thisptr.sampling_rate(i)

    property frame_size:
        def __get__(self): return self.thisptr.frame_size()
        def __set__(self, int i): self.thisptr.frame_size(i)

    property static_frame_size:
        def __get__(self): return self.thisptr.static_frame_size()
        def __set__(self, int i): self.thisptr.static_frame_size(i)

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
        def __set__(self, f):
            raise Exception("NotImplemented")

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


cdef class Synthesis:
    cdef c_Synthesis* thisptr

    def __cinit__(self): self.thisptr = new c_Synthesis()
    def __dealloc__(self): del self.thisptr

    property sampling_rate:
        def __get__(self): return self.thisptr.sampling_rate()
        def __set__(self, int i): self.thisptr.sampling_rate(i)

    property frame_size:
        def __get__(self): return self.thisptr.frame_size()
        def __set__(self, int i): self.thisptr.frame_size(i)

    property hop_size:
        def __get__(self): return self.thisptr.hop_size()
        def __set__(self, int i): self.thisptr.hop_size(i)

    property max_partials:
        def __get__(self): return self.thisptr.max_partials()
        def __set__(self, int i): self.thisptr.max_partials(i)

    def synth_frame(self, Frame frame not None):
        self.thisptr.synth_frame(frame.thisptr)
        return frame.audio

    def synth(self, frames):
        cdef vector[c_Frame*] c_frames
        for frame in frames:
            c_frames.push_back((<Frame>frame).thisptr)
        cdef vector[c_Frame*] output_frames = self.thisptr.synth(c_frames)
        cdef np.ndarray[dtype_t, ndim=1] output = np.zeros(
            output_frames.size() * self.thisptr.hop_size()
        )
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.thisptr.hop_size()
        for i in range(output_frames.size()):
            frame_audio = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, output_frames[i].synth())
            output[i * self.thisptr.hop_size():(i + 1) * self.thisptr.hop_size()] = frame_audio
        return output


cdef class Residual:
    cdef c_Residual* thisptr

    def __cinit__(self): self.thisptr = new c_Residual()
    def __dealloc__(self): del self.thisptr

    property sampling_rate:
        def __get__(self): return self.thisptr.sampling_rate()
        def __set__(self, int i): self.thisptr.sampling_rate(i)

    property frame_size:
        def __get__(self): return self.thisptr.frame_size()
        def __set__(self, int i): self.thisptr.frame_size(i)

    property hop_size:
        def __get__(self): return self.thisptr.hop_size()
        def __set__(self, int i): self.thisptr.hop_size(i)

    def residual_frame(self, np.ndarray[dtype_t, ndim=1] synth,
                       np.ndarray[dtype_t, ndim=1] original):
        cdef np.ndarray[dtype_t, ndim=1] residual = np.zeros(len(synth))
        self.thisptr.residual_frame(len(synth), <double*> synth.data,
                                    len(original), <double*> original.data,
                                    len(residual), <double*> residual.data)
        return residual

    def find_residual(self, np.ndarray[dtype_t, ndim=1] synth,
                      np.ndarray[dtype_t, ndim=1] original):
        cdef np.ndarray[dtype_t, ndim=1] residual = np.zeros(len(synth))
        self.thisptr.find_residual(len(synth), <double*> synth.data,
                                   len(original), <double*> original.data,
                                   len(residual), <double*> residual.data)
        return residual

    def synth_frame(self, Frame frame not None):
        self.thisptr.synth_frame(frame.thisptr)
        return frame.audio

    def synth(self, frames):
        cdef vector[c_Frame*] c_frames
        for frame in frames:
            c_frames.push_back((<Frame>frame).thisptr)
        cdef vector[c_Frame*] output_frames = self.thisptr.synth(c_frames)
        cdef np.ndarray[dtype_t, ndim=1] output = np.zeros(
            output_frames.size() * self.thisptr.hop_size()
        )
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.thisptr.hop_size()
        for i in range(output_frames.size()):
            frame_audio = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, output_frames[i].synth())
            output[i * self.thisptr.hop_size():(i + 1) * self.thisptr.hop_size()] = frame_audio
        return output
