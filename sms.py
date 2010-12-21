# Copyright (c) 2009 John Glover, National University of Ireland, Maynooth
# 
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

import simpl
from simpl import simplsms

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
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")
        self._peaks = simplsms.SMS_SpectralPeaks(self._max_peaks)
        # By default, SMS will change the size of the frames being read depending on the
        # detected fundamental frequency (if any) of the input sound. To prevent this 
        # behaviour (useful when comparing different analysis algorithms), set the
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
            print "set to more than the maximum number of peaks possible in libsms."
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
        num_peaks = simplsms.sms_findPeaks(frame, 
                                           self._analysis_params, 
                                           self._peaks)
        if num_peaks > 0:
            amps = simpl.zeros(num_peaks)
            freqs = simpl.zeros(num_peaks)
            phases = simpl.zeros(num_peaks)
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
        """Find and return all spectral peaks in a given audio signal.
        If the signal contains more than 1 frame worth of audio, it will be broken
        up into separate frames, with a list of peaks returned for each frame."""
        # TODO: This hops by frame size rather than hop size in order to
        #       make sure the results are the same as with libsms. Make sure
        #       we have the same number of frames as the other algorithms.
        self._analysis_params.iSizeSound = len(audio)
        self.peaks = []
        pos = 0
        while pos < len(audio):
            # get the next frame size
            if not self._static_frame_size:
                self.frame_size = self.get_next_frame_size()
            # get the next frame
            frame = audio[pos:pos+self.frame_size]
            # find peaks
            self.peaks.append(self.find_peaks_in_frame(frame))
            pos += self.frame_size
        return self.peaks
    

class SMSPartialTracking(simpl.PartialTracking):
    "Partial tracking using SMS"
    
    def __init__(self):
        simpl.PartialTracking.__init__(self)
        simplsms.sms_init()
        self._analysis_params = simplsms.SMS_AnalParams()
        simplsms.sms_initAnalParams(self._analysis_params)
        self._analysis_params.iSamplingRate = self.sampling_rate
        self._analysis_params.fHighestFreq = 20000
        self._analysis_params.iMaxDelayFrames = 4 # minimum frame delay with libsms
        self._analysis_params.analDelay = 0
        self._analysis_params.minGoodFrames = 1
        self._analysis_params.iCleanTracks = 0
        self._analysis_params.iFormat = simplsms.SMS_FORMAT_HP
        self._analysis_params.nTracks = self.max_partials
        self._analysis_params.nGuides = self.max_partials
        if simplsms.sms_initAnalysis(self._analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")
        self._sms_header = simplsms.SMS_Header()
        simplsms.sms_fillHeader(self._sms_header, self._analysis_params, "simpl")
        self._analysis_frame = simplsms.SMS_Data()
        simplsms.sms_allocFrameH(self._sms_header, self._analysis_frame)
        self.live_partials = [None for i in range(self.max_partials)]
        
    def __del__(self):
        simplsms.sms_freeAnalysis(self._analysis_params)
        simplsms.sms_freeFrame(self._analysis_frame)
        simplsms.sms_free()
        
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
        
    def update_partials(self, frame, frame_number):
        "Streamable (real-time) partial-tracking."
        frame_partials = []
        # load Peak amplitudes, frequencies and phases into arrays
        num_peaks = len(frame)
        amps = simpl.zeros(num_peaks)
        freqs = simpl.zeros(num_peaks)
        phases = simpl.zeros(num_peaks)
        for i in range(num_peaks):
            peak = frame[i]
            amps[i] = peak.amplitude
            freqs[i] = peak.frequency
            phases[i] = peak.phase
        # set peaks in SMS_AnalParams structure
        simplsms.sms_setPeaks(self._analysis_params, amps, freqs, phases)
        # SMS partial tracking
        simplsms.sms_findPartials(self._analysis_frame, self._analysis_params)
        # read values back into amps, freqs, phases
        num_partials = self._analysis_frame.nTracks
        amps = simpl.zeros(num_partials)
        freqs = simpl.zeros(num_partials)
        phases = simpl.zeros(num_partials)
        self._analysis_frame.getSinAmp(amps)
        self._analysis_frame.getSinFreq(freqs)
        self._analysis_frame.getSinPhase(phases)
        # form simpl Partial objects
        for i in range(num_partials):
            # for each partial, if the mag is > 0, this partial is alive
            if amps[i] > 0:
                # create a peak object
                p = simpl.Peak()
                p.amplitude = amps[i]
                p.frequency = freqs[i]
                p.phase = phases[i]
                # add this peak to the appropriate partial
                if not self.live_partials[i]:
                    self.live_partials[i] = simpl.Partial()
                    self.live_partials[i].starting_frame = frame_number
                    self.live_partials[i].partial_number = i
                    self.partials.append(self.live_partials[i])
                self.live_partials[i].add_peak(p)
            # if the mag is 0 and this partial was alive, kill it
            else:
                if self.live_partials[i]:
                    self.live_partials[i] = None
        return frame_partials
    

class SMSSynthesis(simpl.Synthesis):
    "Sinusoidal resynthesis using SMS"
    
    def __init__(self):
        simpl.Synthesis.__init__(self)
        simplsms.sms_init()
        self._synth_params = simplsms.SMS_SynthParams() 
        simplsms.sms_initSynthParams(self._synth_params)
        self._synth_params.iDetSynthType = simplsms.SMS_DET_IFFT
        self._synth_params.iSynthesisType = simplsms.SMS_STYPE_DET
        self._synth_params.iStochasticType = simplsms.SMS_STOC_NONE
        # use the default simpl hop size instead of the default SMS hop size
        self._synth_params.sizeHop = self._hop_size 
        simplsms.sms_initSynth(self._synth_params)
        self._current_frame = simpl.zeros(self.hop_size)
        self._analysis_frame = simplsms.SMS_Data()
        simplsms.sms_allocFrame(self._analysis_frame, self.max_partials, 
                                self.num_stochastic_coeffs, 1, self.stochastic_type, 0)

    def __del__(self):
        simplsms.sms_freeFrame(self._analysis_frame)
        simplsms.sms_freeSynth(self._synth_params)
        simplsms.sms_free()
        
    # properties
    synthesis_type = property(lambda self: self.get_synthesis_type(),
                              lambda self, x: self.set_synthesis_type(x))
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
        self._current_frame = simpl.zeros(self.hop_size)
        
    def get_max_partials(self):
        return self._synth_params.nTracks
        
    def set_max_partials(self, max_partials):
        simplsms.sms_freeSynth(self._synth_params)
        simplsms.sms_freeFrame(self._analysis_frame)
        self._synth_params.nTracks = max_partials
        simplsms.sms_initSynth(self._synth_params)
        simplsms.sms_allocFrame(self._analysis_frame, max_partials, 
                                self.num_stochastic_coeffs, 1, self.stochastic_type, 0)
    
    def get_sampling_rate(self):
        return self._synth_params.iSamplingRate
    
    def set_sampling_rate(self, sampling_rate):
        self._synth_params.iSamplingRate = sampling_rate
    
    def get_synthesis_type(self):
        return self._synth_params.iSynthesisType
    
    def set_synthesis_type(self, synthesis_type):
        self._synth_params.iSynthesisType = synthesis_type
    
    def get_num_stochastic_coeffs(self):
        return self._synth_params.nStochasticCoeff
    
    def set_num_stochastic_coeffs(self, num_stochastic_coeffs):
        self._synth_params.nStochasticCoeff = num_stochastic_coeffs
        simplsms.sms_freeFrame(self._analysis_frame)
        simplsms.sms_allocFrame(self._analysis_frame, self.max_partials, 
                                num_stochastic_coeffs, 1, self.stochastic_type, 0)
    
    def get_stochastic_type(self):
        return self._synth_params.iStochasticType
    
    def set_stochastic_type(self, stochastic_type):
        simplsms.sms_freeSynth(self._synth_params)
        simplsms.sms_freeFrame(self._analysis_frame)
        self._synth_params.iStochasticType = stochastic_type
        simplsms.sms_initSynth(self._synth_params)
        simplsms.sms_allocFrame(self._analysis_frame, self.max_partials, 
                                self.num_stochastic_coeffs, 1, stochastic_type, 0)
        
    def get_original_sampling_rate(self):
        return self._synth_params.iOriginalSRate
    
    def set_original_sampling_rate(self, sampling_rate):
        self._synth_params.iOriginalSRate = sampling_rate
        
    def get_original_hop_size(self):
        return self._synth_params.origSizeHop
    
    def set_original_hop_size(self, hop_size):
        self._synth_params.origSizeHop = hop_size

    def synth_frame(self, peaks):
        "Synthesises a frame of audio, given a list of peaks from tracks"
        amps = simpl.zeros(self.max_partials)
        freqs = simpl.zeros(self.max_partials)
        phases = simpl.zeros(self.max_partials)
        for i in range(len(peaks)):
            p = peaks[i].partial_number
            if p < 0:
                p = i
            amps[p] = peaks[i].amplitude
            freqs[p] = peaks[i].frequency
            phases[p] = peaks[i].phase
        self._analysis_frame.setSinAmp(amps)
        self._analysis_frame.setSinFreq(freqs)
        self._analysis_frame.setSinPha(phases)
        simplsms.sms_synthesize(self._analysis_frame, self._current_frame, self._synth_params)
        return self._current_frame
    

class SMSResidual(simpl.Residual):
    "SMS residual component"
    
    def __init__(self):
        simpl.Residual.__init__(self)
        simplsms.sms_init()
        self._residual_params = simplsms.SMS_ResidualParams()
        simplsms.sms_initResidualParams(self._residual_params)
        self._residual_params.residualSize = self._hop_size# * 2
        simplsms.sms_initResidual(self._residual_params)
        
    def __del__(self):
        simplsms.sms_freeResidual(self._residual_params)
        simplsms.sms_free()
        
    def residual_frame(self, synth, original):
        "Computes the residual signal for a frame of audio"
        simplsms.sms_findResidual(synth, original, self._residual_params)
        residual = simpl.zeros(self._residual_params.residualSize)
        self._residual_params.getResidual(residual)
        return residual

    def synth_frame(self, synth, original):
        "Calculate and return one frame of the synthesised residual signal"
        self.residual_frame(synth, original)
        simplsms.sms_approxResidual(self._residual_params)
        residual_approx = simpl.zeros(self._residual_params.residualSize)
        self._residual_params.getApprox(residual_approx)
        return residual_approx

