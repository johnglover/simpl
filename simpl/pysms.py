import numpy as np
import simpl
import simpl.simplsms as simplsms


class SMSPeakDetection(simpl.PeakDetection):
    "Sinusoidal peak detection using SMS"

    def __init__(self):
        simpl.PeakDetection.__init__(self)
        simplsms.sms_init()
        # analysis parameters
        self._analysis_params = simplsms.SMS_AnalParams()
        simplsms.sms_initAnalParams(self._analysis_params)
        self._analysis_params.iSamplingRate = self._sampling_rate
        # set default hop and frame sizes to match those in the parent class
        self._analysis_params.iFrameRate = self.sampling_rate / self._hop_size
        self._analysis_params.iWindowType = simplsms.SMS_WIN_HAMMING
        self._analysis_params.fHighestFreq = 20000
        self._analysis_params.iMaxDelayFrames = 4
        self._analysis_params.analDelay = 0
        self._analysis_params.minGoodFrames = 1
        self._analysis_params.iCleanTracks = 0
        self._analysis_params.iFormat = simplsms.SMS_FORMAT_HP
        self._analysis_params.nTracks = self._max_peaks
        self._analysis_params.maxPeaks = self._max_peaks
        self._analysis_params.nGuides = self._max_peaks
        self._analysis_params.preEmphasis = 0
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")
        self._peaks = simplsms.SMS_SpectralPeaks(self._max_peaks)
        # By default, SMS will change the size of the frames being read
        # depending on the detected fundamental frequency (if any) of the
        # input sound. To prevent this behaviour (useful when comparing
        # different analysis algorithms), set the
        # _static_frame_size variable to True
        self._static_frame_size = False

    def __del__(self):
        simplsms.sms_freeAnalysis(self._analysis_params)
        simplsms.sms_freeSpectralPeaks(self._peaks)
        simplsms.sms_free()

    # properties
    max_frequency = property(lambda self: self.get_max_frequency(),
                             lambda self, x: self.set_max_frequency(x))
    default_fundamental = property(lambda self: self.get_default_fundamental(),
                                   lambda self, x: self.set_default_fundamental(x))
    max_frame_delay = property(lambda self: self.get_max_frame_delay(),
                               lambda self, x: self.set_max_frame_delay(x))
    analysis_delay = property(lambda self: self.get_analysis_delay(),
                              lambda self, x: self.set_analysis_delay(x))
    min_good_frames = property(lambda self: self.get_min_good_frames(),
                               lambda self, x: self.set_min_good_frames(x))
    min_frequency = property(lambda self: self.get_min_frequency(),
                             lambda self, x: self.set_min_frequency(x))
    min_peak_amp = property(lambda self: self.get_min_peak_amp(),
                            lambda self, x: self.set_min_peak_amp(x))
    clean_tracks = property(lambda self: self.get_clean_tracks(),
                            lambda self, x: self.set_clean_tracks(x))
    format = property(lambda self: self.get_format(),
                      lambda self, x: self.set_format(x))
    pre_emphasis = property(lambda self: self.get_pre_emphasis(),
                            lambda self, x: self.set_pre_emphasis(x))

    def get_max_frequency(self):
        return self._analysis_params.fHighestFreq

    def set_max_frequency(self, max_frequency):
        self._analysis_params.fHighestFreq = max_frequency

    def get_default_fundamental(self):
        return self._analysis_params.fDefaultFundamental

    def set_default_fundamental(self, default_fundamental):
        self._analysis_params.fDefaultFundamental = default_fundamental

    def get_max_frame_delay(self):
        return self._analysis_params.iMaxDelayFrames

    def set_max_frame_delay(self, max_frame_delay):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.iMaxDelayFrames = max_frame_delay
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_analysis_delay(self):
        return self._analysis_params.analDelay

    def set_analysis_delay(self, analysis_delay):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.analDelay = analysis_delay
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_min_good_frames(self):
        return self._analysis_params.minGoodFrames

    def set_min_good_frames(self, min_good_frames):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.minGoodFrames = min_good_frames
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_min_frequency(self):
        return self._analysis_params.fLowestFundamental

    def set_min_frequency(self, min_frequency):
        self._analysis_params.fLowestFundamental = min_frequency
        self._analysis_params.fLowestFreq = min_frequency

    def get_min_peak_amp(self):
        return self._analysis_params.fMinPeakMag

    def set_min_peak_amp(self, min_peak_amp):
        self._analysis_params.fMinPeakMag = min_peak_amp

    def get_clean_tracks(self):
        return self._analysis_params.iCleanTracks

    def set_clean_tracks(self, x):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.iCleanTracks = x
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_format(self):
        return self._analysis_params.iFormat

    def set_format(self, x):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.iFormat = x
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_pre_emphasis(self):
        return self._analysis_params.preEmphasis

    def set_pre_emphasis(self, x):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.preEmphasis = x
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_hop_size(self):
        return self._analysis_params.sizeHop

    def set_hop_size(self, hop_size):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.iFrameRate = self.sampling_rate / hop_size
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_max_peaks(self):
        return self._analysis_params.maxPeaks

    def set_max_peaks(self, max_peaks):
        simplsms.sms_freeAnalysis(self._analysis_params)
        simplsms.sms_freeSpectralPeaks(self._peaks)
        # make sure the new max is less than SMS_MAX_NPEAKS
        if max_peaks > simplsms.SMS_MAX_NPEAKS:
            print "Warning: max peaks (" + str(max_peaks) + ")",
            print "set to more than the max. no. peaks possible in libsms."
            print "         Setting to", simplsms.SMS_MAX_NPEAKS, "instead."
            max_peaks = simplsms.SMS_MAX_NPEAKS
        # set analysis params
        self._max_peaks = max_peaks
        self._analysis_params.nTracks = max_peaks
        self._analysis_params.maxPeaks = max_peaks
        self._analysis_params.nGuides = max_peaks
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")
        # set peaks list
        self._peaks = simplsms.SMS_SpectralPeaks(max_peaks)

    def get_sampling_rate(self):
        return self._analysis_params.iSamplingRate

    def set_sampling_rate(self, sampling_rate):
        self._analysis_params.iSamplingRate = sampling_rate
        simplsms.sms_freeAnalysis(self._analysis_params)
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def set_window_size(self, window_size):
        self._window_size = window_size
        self._analysis_params.iDefaultSizeWindow = window_size

    def get_next_frame_size(self):
        return self._analysis_params.sizeNextRead

    def find_peaks_in_frame(self, frame):
        "Find and return all spectral peaks in a given frame of audio"
        current_peaks = []
        num_peaks = simplsms.sms_findPeaks(frame.audio,
                                           self._analysis_params,
                                           self._peaks)
        if num_peaks > 0:
            amps = np.zeros(num_peaks, dtype=simpl.dtype)
            freqs = np.zeros(num_peaks, dtype=simpl.dtype)
            phases = np.zeros(num_peaks, dtype=simpl.dtype)
            self._peaks.getFreq(freqs)
            self._peaks.getMag(amps)
            self._peaks.getPhase(phases)
            for i in range(num_peaks):
                p = simpl.Peak()
                p.amplitude = amps[i]
                p.frequency = freqs[i]
                p.phase = phases[i]
                current_peaks.append(p)
        return current_peaks

    def find_peaks(self, audio):
        """
        Find and return all spectral peaks in a given audio signal.
        If the signal contains more than 1 frame worth of audio,
        it will be broken up into separate frames, with a list of
        peaks returned for each frame.
        """
        # TODO: This hops by frame size rather than hop size in order to
        #       make sure the results are the same as with libsms. Make sure
        #       we have the same number of frames as the other algorithms.
        self._analysis_params.iSizeSound = len(audio)
        self.frames = []
        pos = 0
        # account for SMS analysis delay
        # need an extra (max_frame_delay - 1) frames
        num_samples = (len(audio) - self.hop_size) + ((self.max_frame_delay - 1) * self.hop_size)
        while pos < num_samples:
            # get the next frame size
            if not self._static_frame_size:
                self.frame_size = self.get_next_frame_size()
            # get the next frame
            frame = simpl.Frame()
            frame.size = self.frame_size
            frame.audio = audio[pos:pos + self.frame_size]
            # find peaks
            frame.peaks = self.find_peaks_in_frame(frame)
            self.frames.append(frame)
            pos += self.frame_size
        return self.frames


class SMSPartialTracking(simpl.PartialTracking):
    "Partial tracking using SMS"

    def __init__(self):
        simpl.PartialTracking.__init__(self)
        simplsms.sms_init()
        self._analysis_params = simplsms.SMS_AnalParams()
        simplsms.sms_initAnalParams(self._analysis_params)
        self._analysis_params.iSamplingRate = self.sampling_rate
        self._analysis_params.fHighestFreq = 20000
        self._analysis_params.iMaxDelayFrames = 4  # libsms minimum
        self._analysis_params.analDelay = 0
        self._analysis_params.minGoodFrames = 1
        self._analysis_params.iCleanTracks = 0
        self._analysis_params.iFormat = simplsms.SMS_FORMAT_HP
        self._analysis_params.nTracks = self._max_partials
        self._analysis_params.nGuides = self._max_partials
        self._analysis_params.preEmphasis = 0
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")
        self._sms_header = simplsms.SMS_Header()
        simplsms.sms_fillHeader(self._sms_header, self._analysis_params, "simpl")
        self._analysis_frame = simplsms.SMS_Data()
        simplsms.sms_allocFrameH(self._sms_header, self._analysis_frame)

    def __del__(self):
        simplsms.sms_freeAnalysis(self._analysis_params)
        simplsms.sms_freeFrame(self._analysis_frame)
        simplsms.sms_free()

    # properties
    max_frequency = property(lambda self: self.get_max_frequency(),
                             lambda self, x: self.set_max_frequency(x))
    default_fundamental = property(lambda self: self.get_default_fundamental(),
                                   lambda self, x: self.set_default_fundamental(x))
    max_frame_delay = property(lambda self: self.get_max_frame_delay(),
                               lambda self, x: self.set_max_frame_delay(x))
    analysis_delay = property(lambda self: self.get_analysis_delay(),
                              lambda self, x: self.set_analysis_delay(x))
    min_good_frames = property(lambda self: self.get_min_good_frames(),
                               lambda self, x: self.set_min_good_frames(x))
    clean_tracks = property(lambda self: self.get_clean_tracks(),
                            lambda self, x: self.set_clean_tracks(x))
    format = property(lambda self: self.get_format(),
                      lambda self, x: self.set_format(x))
    pre_emphasis = property(lambda self: self.get_pre_emphasis(),
                            lambda self, x: self.set_pre_emphasis(x))

    def get_max_frequency(self):
        return self._analysis_params.fHighestFreq

    def set_max_frequency(self, max_frequency):
        self._analysis_params.fHighestFreq = max_frequency

    def get_default_fundamental(self):
        return self._analysis_params.fDefaultFundamental

    def set_default_fundamental(self, default_fundamental):
        self._analysis_params.fDefaultFundamental = default_fundamental

    def get_max_frame_delay(self):
        return self._analysis_params.iMaxDelayFrames

    def set_max_frame_delay(self, max_frame_delay):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.iMaxDelayFrames = max_frame_delay
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_analysis_delay(self):
        return self._analysis_params.analDelay

    def set_analysis_delay(self, x):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.analDelay = x
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_min_good_frames(self):
        return self._analysis_params.minGoodFrames

    def set_min_good_frames(self, x):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.minGoodFrames = x
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_clean_tracks(self):
        return self._analysis_params.iCleanTracks

    def set_clean_tracks(self, x):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.iCleanTracks = x
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_format(self):
        return self._analysis_params.iFormat

    def set_format(self, x):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.iFormat = x
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_pre_emphasis(self):
        return self._analysis_params.preEmphasis

    def set_pre_emphasis(self, x):
        simplsms.sms_freeAnalysis(self._analysis_params)
        self._analysis_params.preEmphasis = x
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")

    def get_max_partials(self):
        return self._analysis_params.nTracks

    def set_max_partials(self, max_partials):
        simplsms.sms_freeAnalysis(self._analysis_params)
        simplsms.sms_freeFrame(self._analysis_frame)
        self._max_partials = max_partials
        self._analysis_params.maxPeaks = max_partials
        self._analysis_params.nTracks = max_partials
        self._analysis_params.nGuides = max_partials
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")
        simplsms.sms_fillHeader(self._sms_header, self._analysis_params, "simpl")
        simplsms.sms_allocFrameH(self._sms_header, self._analysis_frame)

    def update_partials(self, frame):
        "Streamable (real-time) partial-tracking."
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
        # set peaks in SMS_AnalParams structure
        simplsms.sms_setPeaks(self._analysis_params, amps, freqs, phases)
        # SMS partial tracking
        simplsms.sms_findPartials(self._analysis_frame, self._analysis_params)
        # read values back into amps, freqs, phases
        amps = np.zeros(self.max_partials, dtype=simpl.dtype)
        freqs = np.zeros(self.max_partials, dtype=simpl.dtype)
        phases = np.zeros(self.max_partials, dtype=simpl.dtype)
        self._analysis_frame.getSinAmp(amps)
        self._analysis_frame.getSinFreq(freqs)
        self._analysis_frame.getSinPhase(phases)
        peaks = []
        for i in range(self.max_partials):
            p = simpl.Peak()
            p.amplitude = amps[i]
            p.frequency = freqs[i]
            p.phase = phases[i]
            peaks.append(p)
        return peaks

    def find_partials(self, frames):
        """Find partials from the sinusoidal peaks in a list of Frames"""
        self.frames = []
        for frame in frames:
            frame.partials = self.update_partials(frame)
            self.frames.append(frame)
        # account for SMS analysis delay
        # the first extra (max_frame_delay) frames are blank
        if len(self.frames) > (self.max_frame_delay):
            self.frames = self.frames[self.max_frame_delay:]
        return self.frames


class SMSSynthesis(simpl.Synthesis):
    "Sinusoidal resynthesis using SMS"

    def __init__(self):
        simpl.Synthesis.__init__(self)
        simplsms.sms_init()
        self._synth_params = simplsms.SMS_SynthParams()
        simplsms.sms_initSynthParams(self._synth_params)
        self._synth_params.iSamplingRate = self._sampling_rate
        self._synth_params.iDetSynthType = simplsms.SMS_DET_SIN
        self._synth_params.iSynthesisType = simplsms.SMS_STYPE_DET
        self._synth_params.iStochasticType = simplsms.SMS_STOC_NONE
        self._synth_params.sizeHop = self._hop_size
        self._synth_params.nTracks = self._max_partials
        self._synth_params.deEmphasis = 0
        simplsms.sms_initSynth(self._synth_params)
        self._current_frame = np.zeros(self._hop_size, dtype=simpl.dtype)
        self._analysis_frame = simplsms.SMS_Data()
        simplsms.sms_allocFrame(self._analysis_frame, self.max_partials,
                                self.num_stochastic_coeffs, 1,
                                self.stochastic_type, 0)

    def __del__(self):
        simplsms.sms_freeFrame(self._analysis_frame)
        simplsms.sms_freeSynth(self._synth_params)
        simplsms.sms_free()

    # properties
    synthesis_type = property(lambda self: self.get_synthesis_type(),
                              lambda self, x: self.set_synthesis_type(x))
    det_synthesis_type = property(lambda self: self.get_det_synthesis_type(),
                                  lambda self, x: self.set_det_synthesis_type(x))
    num_stochastic_coeffs = property(lambda self: self.get_num_stochastic_coeffs(),
                                     lambda self, x: self.set_num_stochastic_coeffs(x))
    stochastic_type = property(lambda self: self.get_stochastic_type(),
                               lambda self, x: self.set_stochastic_type(x))
    original_sampling_rate = property(lambda self: self.get_original_sampling_rate(),
                                      lambda self, x: self.set_original_sampling_rate(x))
    original_hop_size = property(lambda self: self.get_original_hop_size(),
                                 lambda self, x: self.set_original_hop_size(x))

    def get_hop_size(self):
        return self._synth_params.sizeHop

    def set_hop_size(self, hop_size):
        simplsms.sms_freeSynth(self._synth_params)
        self._synth_params.sizeHop = hop_size
        simplsms.sms_initSynth(self._synth_params)
        self._current_frame = np.zeros(hop_size, dtype=simpl.dtype)

    def get_max_partials(self):
        return self._synth_params.nTracks

    def set_max_partials(self, max_partials):
        simplsms.sms_freeSynth(self._synth_params)
        simplsms.sms_freeFrame(self._analysis_frame)
        self._synth_params.nTracks = max_partials
        simplsms.sms_initSynth(self._synth_params)
        simplsms.sms_allocFrame(self._analysis_frame, max_partials,
                                self.num_stochastic_coeffs, 1,
                                self.stochastic_type, 0)

    def get_sampling_rate(self):
        return self._synth_params.iSamplingRate

    def set_sampling_rate(self, sampling_rate):
        self._synth_params.iSamplingRate = sampling_rate

    def get_synthesis_type(self):
        return self._synth_params.iSynthesisType

    def set_synthesis_type(self, synthesis_type):
        self._synth_params.iSynthesisType = synthesis_type

    def get_det_synthesis_type(self):
        return self._synth_params.iDetSynthesisType

    def set_det_synthesis_type(self, det_synthesis_type):
        self._synth_params.iDetSynthType = det_synthesis_type

    def get_num_stochastic_coeffs(self):
        return self._synth_params.nStochasticCoeff

    def set_num_stochastic_coeffs(self, num_stochastic_coeffs):
        self._synth_params.nStochasticCoeff = num_stochastic_coeffs
        simplsms.sms_freeFrame(self._analysis_frame)
        simplsms.sms_allocFrame(self._analysis_frame, self.max_partials,
                                num_stochastic_coeffs, 1,
                                self.stochastic_type, 0)

    def get_stochastic_type(self):
        return self._synth_params.iStochasticType

    def set_stochastic_type(self, stochastic_type):
        simplsms.sms_freeSynth(self._synth_params)
        simplsms.sms_freeFrame(self._analysis_frame)
        self._synth_params.iStochasticType = stochastic_type
        simplsms.sms_initSynth(self._synth_params)
        simplsms.sms_allocFrame(self._analysis_frame, self.max_partials,
                                self.num_stochastic_coeffs, 1,
                                stochastic_type, 0)

    def get_original_sampling_rate(self):
        return self._synth_params.iOriginalSRate

    def set_original_sampling_rate(self, sampling_rate):
        self._synth_params.iOriginalSRate = sampling_rate

    def get_original_hop_size(self):
        return self._synth_params.origSizeHop

    def set_original_hop_size(self, hop_size):
        self._synth_params.origSizeHop = hop_size

    def synth_frame(self, frame):
        "Synthesises a frame of audio"
        amps = np.zeros(self.max_partials, dtype=simpl.dtype)
        freqs = np.zeros(self.max_partials, dtype=simpl.dtype)
        phases = np.zeros(self.max_partials, dtype=simpl.dtype)
        num_partials = min(self.max_partials, len(frame.partials))
        for i in range(num_partials):
            amps[i] = frame.partials[i].amplitude
            freqs[i] = frame.partials[i].frequency
            phases[i] = frame.partials[i].phase
        self._analysis_frame.setSinAmp(amps)
        self._analysis_frame.setSinFreq(freqs)
        self._analysis_frame.setSinPha(phases)
        simplsms.sms_synthesize(self._analysis_frame,
                                self._current_frame,
                                self._synth_params)
        return self._current_frame


class SMSResidual(simpl.Residual):
    "SMS residual component"

    def __init__(self):
        simpl.Residual.__init__(self)
        simplsms.sms_init()
        self._residual_params = simplsms.SMS_ResidualParams()
        simplsms.sms_initResidualParams(self._residual_params)
        self._residual_params.hopSize = self._hop_size
        simplsms.sms_initResidual(self._residual_params)

    def __del__(self):
        simplsms.sms_freeResidual(self._residual_params)
        simplsms.sms_free()

    def get_hop_size(self):
        return self._residual_params.hopSize

    def set_hop_size(self, hop_size):
        simplsms.sms_freeResidual(self._residual_params)
        self._residual_params.hopSize = hop_size
        simplsms.sms_initResidual(self._residual_params)

    def residual_frame(self, synth, original):
        "Computes the residual signal for a frame of audio"
        simplsms.sms_findResidual(synth, original, self._residual_params)
        residual = np.zeros(self._residual_params.hopSize, dtype=simpl.dtype)
        self._residual_params.getResidual(residual)
        return residual

    def find_residual(self, synth, original):
        "Calculate and return the residual signal"
        num_frames = len(original) / self.hop_size
        residual = np.array([], dtype=simpl.dtype)
        sample_offset = 0

        for i in range(num_frames):
            synth_frame = synth[sample_offset:sample_offset + self.hop_size]
            original_frame = original[sample_offset:sample_offset + self.hop_size]
            residual = np.hstack((
                residual, self.residual_frame(synth_frame, original_frame)
            ))
            sample_offset += self.hop_size
        return residual

    def synth_frame(self, synth, original):
        "Calculate and return one frame of the synthesised residual signal"
        residual = self.residual_frame(synth, original)
        approx = np.zeros(self._residual_params.hopSize, dtype=simpl.dtype)
        simplsms.sms_approxResidual(residual, approx, self._residual_params)
        return approx
