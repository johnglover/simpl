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
import pysndobj
import numpy as np

class SndObjPeakDetection(simpl.PeakDetection):
    "Sinusoidal peak detection using the SndObj library (Instantaneous Frequency)"
    def __init__(self):
        simpl.PeakDetection.__init__(self)
        self._input = pysndobj.SndObj()
        self._input.SetVectorSize(self.frame_size)
        self._window = pysndobj.HammingTable(self.frame_size, 0.5)
        self._ifgram = pysndobj.IFGram(self._window, self._input, 1, 
                                         self.frame_size, self.hop_size)
        self._threshold = 0.003
        self._analysis = pysndobj.SinAnal(self._ifgram, self._threshold,
                                             self.max_peaks)
        
    # properties
    threshold = property(lambda self: self.get_threshold(),
                         lambda self, x: self.set_threshold(x))
        
    def set_frame_size(self, frame_size):
        "Set the analysis frame size"
        self._input.SetVectorSize(frame_size)
        if self.window_type == "hamming":
            self._window = pysndobj.HammingTable(frame_size, 0.5)
        elif self.window_type >=0 and self.window_type <= 1:
            self._window = pysndobj.HammingTable(frame_size, self.window_type)
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
            self._window = pysndobj.HammingTable(self.frame_size, 0.5)
        elif window_type >=0 and window_type <= 1:
            self._window = pysndobj.HammingTable(self.frame_size, window_type)
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
        current_peaks = []
        self._input.PushIn(frame)
        self._input.DoProcess()
        self._ifgram.DoProcess()
        num_peaks_found = self._analysis.FindPeaks()
        # loop through analysis output and create peak objects
        for i in range(num_peaks_found):
            p = simpl.Peak()
            p.amplitude = self._analysis.Output(i*3)
            p.frequency = self._analysis.Output((i*3)+1)
            p.phase = self._analysis.Output((i*3)+2)
            current_peaks.append(p)
        return current_peaks
    

class SndObjPartialTracking(simpl.PartialTracking):
    "Partial tracking using the algorithm from the Sound Object Library"
    def __init__(self):
        simpl.PartialTracking.__init__(self)
        self._threshold = 0.003 # todo: property
        self._num_bins = 1025 # todo: property
        self._analysis = pysndobj.SinAnal(pysndobj.SndObj(), self._num_bins,
                                             self._threshold, self.max_partials)
        
    def set_max_partials(self, num_partials):
        self._analysis.Set("max tracks", num_partials)
        self._max_partials = num_partials
             
    def update_partials(self, frame, frame_number):
        "Streamable (real-time) partial-tracking."
        frame_partials = []
        # load Peak amplitudes, frequencies and phases into arrays
        num_peaks = len(frame)
        amps = np.zeros(num_peaks, dtype=np.float32)
        freqs = np.zeros(num_peaks, dtype=np.float32)
        phases = np.zeros(num_peaks, dtype=np.float32)
        for i in range(num_peaks):
            peak = frame[i]
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
            peak.amplitude = self._analysis.Output(i*3)
            peak.frequency = self._analysis.Output((i*3)+1)
            peak.phase = self._analysis.Output((i*3)+2)
            id = self._analysis.GetTrackID(i)
            # if this is a continuing partial, create a peak and append it
            if id >= 0 and id <= len(self.partials) - 1:
                self.partials[id].add_peak(peak)
            # if not, make a new partial
            else:
                partial = simpl.Partial()
                partial.starting_frame = frame_number
                partial.add_peak(peak)
                self.partials.append(partial)
            frame_partials.append(peak)
        return frame_partials
    
                
class SimplSndObjAnalysisWrapper(pysndobj.SinAnal):
    """An object that takes simpl Peaks and presents them as SndObj analysis 
    data to the SndObj synthesis objects."""
    def __init__(self):
        pysndobj.SinAnal.__init__(self)
        self.peaks = None
        
    def GetTracks(self):
        return len(self.peaks)

    def GetTrackID(self, partial_number):
        return self.peaks[partial_number].partial_id
        
    def Output(self, position):
        peak = int(position) / 3
        data_field = int(position) % 3
        if data_field is 0:
            return self.peaks[peak].amplitude
        elif data_field is 1:
            return self.peaks[peak].frequency
        elif data_field is 2:
            return self.peaks[peak].phase
        
        
class SndObjSynthesis(simpl.Synthesis):
    "Sinusoidal resynthesis using the SndObj library"
    def __init__(self, synthesis_type='adsyn'):
        simpl.Synthesis.__init__(self)
        self._analysis = SimplSndObjAnalysisWrapper()
        self._table = pysndobj.HarmTable(10000, 1, 1, 0.25)
        if synthesis_type == 'adsyn':
            self._synth = pysndobj.AdSyn(self._analysis, self.max_partials,
                                            self._table, 1, 1, self.hop_size)
        elif synthesis_type == 'sinsyn':
            self._synth = pysndobj.SinSyn(self._analysis, self.max_partials,
                                          self._table, 1, self.hop_size)
        else:
            raise Exception("UnknownSynthesisType")
        self._current_frame = np.zeros(self.hop_size, dtype=np.float32)
        
    def set_hop_size(self, hop_size):
        self._synth.SetVectorSize(hop_size)
        self._hop_size = hop_size
        self._current_frame = np.zeros(hop_size, dtype=np.float32)
        
    def set_max_partials(self, num_partials):
        self._synth.Set('max tracks', num_partials)
        self._max_partials = num_partials
        
    def _synth_frame(self, peaks):
        "Synthesises a frame of audio, given a list of peaks from tracks"
        self._analysis.peaks = peaks
        if len(peaks) > self._max_partials:
            self._synth.Set('max tracks', len(peaks))
            self._max_partials = len(peaks)
        self._synth.DoProcess()
        self._synth.PopOut(self._current_frame)
        return self._current_frame

    def synth(self, partials, original=None):
        "Synthesise audio from the given partials"
        # return an empty frame if there are no partials
        if not partials:
            return simpl.array([])
        
        audio_out = simpl.array([])
        current_partials = []
        num_frames = max([partial.get_last_frame() for partial in partials])

        # for each frame of audio
        for frame_number in range(num_frames):
            # get all partials that start on this frame, append to list of continuing partials
            current_partials.extend([partial.list_peaks() for partial in partials if partial.starting_frame == frame_number])    
            # get all peaks to be synthesised for this frame
            current_peaks = []
            for partial_number, partial in enumerate(current_partials):
                try:
                    current_peaks.append(partial.next())
                except StopIteration:
                    # End of partial. Set this partial to None, remove it from the list later
                    current_partials[partial_number] = None
                    
            # synth frame
            audio_out = np.hstack((audio_out, self._synth_frame(current_peaks)))
            # remove any finished partials
            current_partials = [partial for partial in current_partials if partial]
            
        return audio_out