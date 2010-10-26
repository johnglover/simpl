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
import simpl.pysms as pysms

class SMSPeakDetection(simpl.PeakDetection):
    "Sinusoidal peak detection using SMS"
    _instances = 0
    
    def __init__(self):
        # limit this to only 1 instance at a time as calls to libsms are not independent,
        # some static C variables are used. These should really be addressed in libsms.
        # todo: silently treat this as a Singleton object rather than raising an exception?
        SMSPeakDetection._instances += 1
        if SMSPeakDetection._instances > 1:
            raise Exception("Currently only 1 instance of each SMS analysis/synthesis object can exist at once")
        simpl.PeakDetection.__init__(self)
        pysms.sms_init()
        # analysis parameters
        self._analysis_params = pysms.SMS_AnalParams()
        self._analysis_params.iSamplingRate = self.sampling_rate
        self._analysis_params.iWindowType = pysms.SMS_WIN_HAMMING
        self._analysis_params.fHighestFreq = 20000
        #self._analysis_params.fLowestFundamental = 50
        #self._analysis_params.fDefaultFundamental = 100
        self._analysis_params.iMaxDelayFrames = 4
        self._analysis_params.analDelay = 0
        self._analysis_params.minGoodFrames = 1
        self._analysis_params.iFormat = pysms.SMS_FORMAT_HP
        pysms.sms_initAnalysis(self._analysis_params)
        self._peaks = pysms.SMS_SpectralPeaks(self.max_peaks)
        # By default, SMS will change the size of the frames being read depending on the
        # detected fundamental frequency (if any) of the input sound. To prevent this 
        # behaviour (useful when comparing different analysis algorithms), set the
        # static_frame_size variable to True
        self.static_frame_size = False
        #self.static_frame_size = True
        # set default hop and frame sizes to match those in the parent class
        self._analysis_params.iFrameRate = self.sampling_rate / self._hop_size
        pysms.sms_changeHopSize(self._hop_size, self._analysis_params)
        self._analysis_params.fDefaultFundamental = float(self.sampling_rate)/self.frame_size
        self._analysis_params.fLowestFundamental = self._analysis_params.fDefaultFundamental

        
    def __del__(self):
        pysms.sms_freeAnalysis(self._analysis_params)
        pysms.sms_free()
        SMSPeakDetection._instances -= 1
        
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
        self._analysis_params.peakParams.fHighestFreq = max_frequency
        
    def get_default_fundamental(self):
        return self._analysis_params.fDefaultFundamental
    
    def set_default_fundamental(self, default_fundamental):
        self._analysis_params.fDefaultFundamental = default_fundamental
        
    def get_max_frame_delay(self):
        return self._analysis_params.iMaxDelayFrames
    
    def set_max_frame_delay(self, max_frame_delay):
        self._analysis_params.iMaxDelayFrames = max_frame_delay
        
    def get_analysis_delay(self):
        return self._analysis_params.analDelay
    
    def set_analysis_delay(self, analysis_delay):
        self._analysis_params.analDelay = analysis_delay
        
    def get_min_good_frames(self):
        return self._analysis_params.minGoodFrames
    
    def set_min_good_frames(self, min_good_frames):
        self._analysis_params.minGoodFrames = min_good_frames
        
    def get_min_frequency(self):
        return self._analysis_params.fLowestFundamental
    
    def set_min_frequency(self, min_frequency):
        self._analysis_params.fLowestFundamental = min_frequency
        self._analysis_params.peakParams.fLowestFreq = min_frequency
        
    def get_min_peak_amp(self):
        return self._analysis_params.fMinPeakMag
    
    def set_min_peak_amp(self, min_peak_amp):
        self._analysis_params.fMinPeakMag = min_peak_amp
        self._analysis_params.peakParams.fMinPeakMag = min_peak_amp
    
    def get_hop_size(self):
        return self._analysis_params.sizeHop
     
    def set_hop_size(self, hop_size):
        self._analysis_params.iFrameRate = self.sampling_rate / hop_size
        pysms.sms_changeHopSize(hop_size, self._analysis_params)
        
    def set_max_peaks(self, max_peaks):
        # todo: compare to SMS_MAX_NPEAKS?
        self._max_peaks = max_peaks
        self._analysis_params.peakParams.iMaxPeaks = max_peaks
        self._peaks = pysms.SMS_SpectralPeaks(max_peaks)
    
    def set_sampling_rate(self, sampling_rate):
        self._sampling_rate = sampling_rate
        self._analysis_params.iSamplingRate = sampling_rate
        
    def set_window_size(self, window_size):
        self._window_size = window_size
        self._analysis_params.iDefaultSizeWindow = window_size
        
    def find_peaks_in_frame(self, frame):
        "Find and return all spectral peaks in a given frame of audio"
        current_peaks = []
        num_peaks = pysms.sms_findPeaks(frame, 
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
        self.peaks = []
        pos = 0
        self._analysis_params.iSizeSound = len(audio)
        while pos < len(audio):
            # change frame size based on sizeNextRead
            if not self.static_frame_size:
                if pos + self.frame_size < len(audio):
                    self.frame_size = self._analysis_params.sizeNextRead
                else:
                    self.frame_size = len(audio) - pos
            # get the next frame
            frame = audio[pos:pos+self.frame_size]
            # find peaks
            self.peaks.append(self.find_peaks_in_frame(frame))
            pos += self.hop_size
        return self.peaks
    

class SMSPartialTracking(simpl.PartialTracking):
    "Partial tracking using SMS"
    _instances = 0
    
    def __init__(self):
        # limit this to only 1 instance at a time as calls to libsms are not independent,
        # some static C variables are used. These should really be addressed in libsms.
        SMSPartialTracking._instances += 1
        if SMSPartialTracking._instances > 1:
            raise Exception("Currently only 1 instance of each SMS analysis/synthesis object can exist at once")
        simpl.PartialTracking.__init__(self)
        pysms.sms_init()
        self._analysis_params = pysms.SMS_AnalParams()
        self._analysis_params.iSamplingRate = self.sampling_rate
        self._analysis_params.fHighestFreq = 20000
        self._analysis_params.fLowestFundamental = 50
        self._analysis_params.fDefaultFundamental = 100
        self._analysis_params.iMaxDelayFrames = 3 # minimum frame delay with libsms
        self._analysis_params.analDelay = 0
        self._analysis_params.minGoodFrames = 1
        self._analysis_params.iFormat = pysms.SMS_FORMAT_HP
        self._analysis_params.nTracks = self.max_partials
        self._analysis_params.nGuides = self.max_partials
        pysms.sms_initAnalysis(self._analysis_params)
        self._analysis_frame = pysms.SMS_Data()
        self.live_partials = [None for i in range(self.max_partials)]
        
    def __del__(self):
        pysms.sms_freeAnalysis(self._analysis_params)
        pysms.sms_free()
        SMSPartialTracking._instances -= 1
        
    def set_max_partials(self, max_partials):
        self._max_partials = max_partials
        self._analysis_params.nTracks = max_partials
        self._analysis_params.nGuides = max_partials
        
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
        pysms.sms_setPeaks(self._analysis_params, amps, freqs, phases)
        # SMS partial tracking
        pysms.sms_findPartials(self._analysis_frame, self._analysis_params)
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
                    self.partials.append(self.live_partials[i])
                self.live_partials[i].add_peak(p)
            # if the mag is 0 and this partial was alive, kill it
            else:
                if self.live_partials[i]:
                    self.live_partials[i] = None
        return frame_partials
    

class SMSSynthesis(simpl.Synthesis):
    "Sinusoidal resynthesis using SMS"
    _instances = 0
    
    def __init__(self):
        SMSSynthesis._instances += 1
        if SMSSynthesis._instances > 1:
            raise Exception("Currently only 1 instance of each SMS analysis/synthesis object can exist at once")   
        simpl.Synthesis.__init__(self)
        pysms.sms_init()
        self._synth_params = pysms.SMS_SynthParams() 
        self._synth_params.iDetSynthType = pysms.SMS_DET_SIN
        # use the default simpl hop size instead of the default SMS hop size
        self._synth_params.sizeHop = self._hop_size 
        pysms.sms_initSynth(self._synth_params)
        self._current_frame = simpl.zeros(self.hop_size)
        self._analysis_frame = pysms.SMS_Data()
        pysms.sms_allocFrame(self._analysis_frame, self.max_partials, 
                             self.num_stochastic_coeffs, 1, self.stochastic_type, 0)

    def __del__(self):
        pysms.sms_freeFrame(self._analysis_frame)
        pysms.sms_freeSynth(self._synth_params)
        pysms.sms_free()
        SMSSynthesis._instances -= 1
        
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
        self._synth_params.sizeHop = hop_size
        self._current_frame = simpl.zeros(self.hop_size)
        
    def get_max_partials(self):
        return self._synth_params.nTracks
        
    def set_max_partials(self, max_partials):
        self._synth_params.nTracks = max_partials
        pysms.sms_allocFrame(self._analysis_frame, max_partials, 
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
        pysms.sms_allocFrame(self._analysis_frame, self.max_partials, 
                             num_stochastic_coeffs, 1, self.stochastic_type, 0)
    
    def get_stochastic_type(self):
        return self._synth_params.iStochasticType
    
    def set_stochastic_type(self, stochastic_type):
        self._synth_params.iStochasticType = stochastic_type
        pysms.sms_allocFrame(self._analysis_frame, self.max_partials, 
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
        for i in range(self.max_partials):
            if i < len(peaks):
                amps[i] = peaks[i].amplitude
                freqs[i] = peaks[i].frequency
                phases[i] = peaks[i].phase
        self._analysis_frame.setSinAmp(amps)
        self._analysis_frame.setSinFreq(freqs)
        self._analysis_frame.setSinPha(phases)
        pysms.sms_synthesize(self._analysis_frame, self._current_frame, self._synth_params)
        return self._current_frame
    

class SMSResidual(simpl.Residual):
    _instances = 0
    
    def __init__(self):
        SMSResidual._instances += 1
        if SMSResidual._instances > 1:
            raise Exception("Currently only 1 instance of each SMS analysis/synthesis object can exist at once")
        simpl.Residual.__init__(self)
        pysms.sms_init()
        self._analysis_params = pysms.SMS_AnalParams()
        pysms.sms_initAnalysis(self._analysis_params)
        
    def __del__(self):
        pysms.sms_free()
        SMSSynthesis._instances -= 1
        
    def find_residual(self, synth, original):
        "Calculate and return the residual signal"
        residual = simpl.zeros(synth.size)
        if pysms.sms_findResidual(synth, original, residual, self._analysis_params) == -1:
            raise Exception("Residual error: Synthesised audio and original audio have different lengths")
        return residual

