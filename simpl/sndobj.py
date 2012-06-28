import simpl
import simpl.simplsndobj as simplsndobj
import numpy as np


class SndObjPeakDetection(simpl.PeakDetection):
    "Sinusoidal peak detection using the SndObj library"
    def __init__(self):
        simpl.PeakDetection.__init__(self)
        self._input = simplsndobj.SndObj()
        self._input.SetVectorSize(self.frame_size)
        self._window = simplsndobj.HammingTable(self.frame_size, 0.5)
        self._ifgram = simplsndobj.IFGram(self._window, self._input, 1,
                                         self.frame_size, self.hop_size)
        self._threshold = 0.003
        self._analysis = simplsndobj.SinAnal(self._ifgram, self._threshold,
                                             self.max_peaks)

    # properties
    threshold = property(lambda self: self.get_threshold(),
                         lambda self, x: self.set_threshold(x))

    def set_frame_size(self, frame_size):
        "Set the analysis frame size"
        self._input.SetVectorSize(frame_size)
        if self.window_type == "hamming":
            self._window = simplsndobj.HammingTable(frame_size, 0.5)
        elif self.window_type >= 0 and self.window_type <= 1:
            self._window = simplsndobj.HammingTable(frame_size, self.window_type)
        self._ifgram.Connect("window", self._window)
        self._ifgram.Set("fft size", frame_size)
        self._frame_size = frame_size

    def set_hop_size(self, hop_size):
        self._ifgram.Set("hop size", hop_size)
        self._hop_size = hop_size

    def set_max_peaks(self, max_peaks):
        "Set the maximum number of peaks detected"
        self._analysis.Set("max tracks", max_peaks)
        self._max_peaks = max_peaks

    def set_window_type(self, window_type):
        "Set the analysis window type"
        if window_type == "hamming":
            self._window = simplsndobj.HammingTable(self.frame_size, 0.5)
        elif window_type >= 0 and window_type <= 1:
            self._window = simplsndobj.HammingTable(self.frame_size, window_type)
        else:
            raise Exception("UnknownWindowType")
        self._ifgram.Connect("window", self._window)
        self._window_type = window_type

    def get_threshold(self):
        return self._threshold

    def set_threshold(self, threshold):
        self._analysis.Set("threshold", threshold)
        self._threshold = threshold

    def find_peaks_in_frame(self, frame):
        "Find and return all spectral peaks in a given frame of audio"
        peaks = []
        self._input.PushIn(frame.audio)
        self._input.DoProcess()
        self._ifgram.DoProcess()
        num_peaks_found = self._analysis.FindPeaks()

        # loop through analysis output and create peak objects
        for i in range(num_peaks_found):
            p = simpl.Peak()
            p.amplitude = self._analysis.Output(i * 3)
            p.frequency = self._analysis.Output((i * 3) + 1)
            p.phase = self._analysis.Output((i * 3) + 2)
            if not peaks:
                peaks.append(p)
            else:
                if np.abs(p.frequency - peaks[-1].frequency) > self._min_peak_separation:
                    peaks.append(p)
                else:
                    if p.amplitude > peaks[-1].amplitude:
                        peaks.remove(peaks[-1])
                        peaks.append(p)
        return peaks


class SndObjPartialTracking(simpl.PartialTracking):
    "Partial tracking using the algorithm from the Sound Object Library"
    def __init__(self):
        simpl.PartialTracking.__init__(self)
        self._threshold = 0.003
        self._num_bins = 1025
        self._analysis = simplsndobj.SinAnal(
            simplsndobj.SndObj(), self._num_bins,
            self._threshold, self.max_partials
        )

    def set_max_partials(self, num_partials):
        self._analysis.Set("max tracks", num_partials)
        self._max_partials = num_partials

    def update_partials(self, frame):
        "Streamable (real-time) partial-tracking."
        partials = []
        # load Peak amplitudes, frequencies and phases into arrays
        num_peaks = len(frame.peaks)
        amps = np.zeros(num_peaks, dtype=simpl.dtype)
        freqs = np.zeros(num_peaks, dtype=simpl.dtype)
        phases = np.zeros(num_peaks, dtype=simpl.dtype)
        for i in range(num_peaks):
            peak = frame.peaks[i]
            amps[i] = peak.amplitude
            freqs[i] = peak.frequency
            phases[i] = peak.phase
        # set peaks in SndObj SinAnal object
        self._analysis.SetPeaks(amps, freqs, phases)
        # call SndObj partial tracking
        self._analysis.PartialTracking()
        # form Partial objects
        num_partials = self._analysis.GetTracks()
        for i in range(num_partials):
            peak = simpl.Peak()
            peak.amplitude = self._analysis.Output(i * 3)
            peak.frequency = self._analysis.Output((i * 3) + 1)
            peak.phase = self._analysis.Output((i * 3) + 2)
            partials.append(peak)
        for i in range(num_partials, self.max_partials):
            peak = simpl.Peak()
            partials.append(peak)
        return partials


class SimplSndObjAnalysisWrapper(simplsndobj.SinAnal):
    """An object that takes simpl Peaks and presents them as SndObj analysis
    data to the SndObj synthesis objects."""
    def __init__(self):
        simplsndobj.SinAnal.__init__(self)
        self.partials = []

    def GetTracks(self):
        return len(self.partials)

    def GetTrackID(self, partial_number):
        if partial_number < len(self.partials):
            return partial_number
        else:
            # TODO: what should this return if no matching partial found?
            return 0

    def Output(self, position):
        peak = int(position) / 3
        if peak > len(self.partials):
            # TODO: what should this return if no matching partial found?
            return 0.0

        data_field = int(position) % 3
        if data_field is 0:
            return self.partials[peak].amplitude
        elif data_field is 1:
            return self.partials[peak].frequency
        elif data_field is 2:
            return self.partials[peak].phase


class SndObjSynthesis(simpl.Synthesis):
    "Sinusoidal resynthesis using the SndObj library"
    def __init__(self, synthesis_type='adsyn'):
        simpl.Synthesis.__init__(self)
        self._analysis = SimplSndObjAnalysisWrapper()
        self._table = simplsndobj.HarmTable(10000, 1, 1, 0.25)
        if synthesis_type == 'adsyn':
            self._synth = simplsndobj.AdSyn(self._analysis, self.max_partials,
                                            self._table, 1, 1, self.hop_size)
        elif synthesis_type == 'sinsyn':
            self._synth = simplsndobj.SinSyn(self._analysis, self.max_partials,
                                          self._table, 1, self.hop_size)
        else:
            raise Exception("UnknownSynthesisType")
        self._current_frame = np.zeros(self.hop_size, dtype=np.dtype)

    def set_hop_size(self, hop_size):
        self._synth.SetVectorSize(hop_size)
        self._hop_size = hop_size
        self._current_frame = np.zeros(hop_size, dtype=np.dtype)

    def set_max_partials(self, num_partials):
        self._synth.Set('max tracks', num_partials)
        self._max_partials = num_partials

    def synth_frame(self, frame):
        "Synthesises a frame of audio, given a list of peaks from tracks"
        self._analysis.partials = frame.partials
        if len(frame.partials) > self._max_partials:
            self.max_partials = len(frame.partials)
        self._synth.DoProcess()
        self._synth.PopOut(self._current_frame)
        return self._current_frame
