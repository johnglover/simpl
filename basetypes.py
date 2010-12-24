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
import numpy as np

class Peak(object):
    "A spectral peak"
    def __init__(self):
        self.amplitude = 0.0
        self.frequency = 0.0
        self.phase = 0.0
        self.next_peak = None
        self.previous_peak = None
        self.partial_id = None
        self.partial_position = None
        self.frame_number = None
        
    def is_start_of_partial(self):
        return self.previous_peak is None
        
    def is_free(self, direction='forwards'):
        "Returns true iff this peak is unmatched in the given direction, and has positive amplitude"
        if self.amplitude <= 0:
            return False
        if direction == 'forwards':
            if self.next_peak:
                return False
        elif direction == 'backwards':
            if self.previous_peak:
                return False
        else:
            return False
        return True
    
    
def compare_peak_amps(peak_x, peak_y):
    """Compares two peaks, and returns 1, 0 or -1 if the first has a greater
    amplitude than the second, they have the same amplitude, or the second has
    a greater amplitude than the first respectively.
    Can be used to sort lists of peaks."""
    if peak_x.amplitude > peak_y.amplitude:
        return 1
    elif peak_x.amplitude < peak_y.amplitude:
        return -1
    else:
        return 0
        
def compare_peak_freqs(peak_x, peak_y):
    """Compares two peaks, and returns 1, 0 or -1 if the first has a greater
    frequency than the second, they have the same frequency, or the second has
    a greater frequency than the first respectively.
    Can be used to sort lists of peaks."""
    if peak_x.frequency > peak_y.frequency:
        return 1
    elif peak_x.frequency < peak_y.frequency:
        return -1
    else:
        return 0


class Partial(object):
    "Represents a sinuoidal partial or track, an ordered sequence of Peaks"    
    _num_partials = 0
    
    def __init__(self):
        "Initialise peaks list and increment partial_id"
        self.peaks = []
        self.starting_frame = 0
        self.partial_number = -1
        self.partial_id = Partial._num_partials
        Partial._num_partials += 1
    
    def add_peak(self, peak):
        "Add peak to this partial, setting its id and partial_id."
        partial_position = len(self.peaks)
        last_peak = self.get_last_peak()
        self.peaks.append(peak)
        if last_peak:
            last_peak.next_peak = peak
            peak.previous_peak = last_peak
        peak.partial_position = partial_position
        peak.partial_id = self.partial_id
        peak.partial_number = self.partial_number
        
    def get_length(self):
        "Return the length of this partial (as a number of frames)"
        return len(self.peaks)
        
    def get_last_frame(self):
        "Return the frame number of the last frame in this partial"
        return self.starting_frame + self.get_length()
        
    def get_last_peak(self):
        "Return the last peak of this partial"
        if self.peaks:
            return self.peaks[-1]
        return None
        
    def list_peaks(self):
        "A generator that returns the peaks in this partial"
        for peak in self.peaks:
            yield peak


class Frame(object):
    """Represents a frame of audio information.
    This can be: - raw audio samples 
                 - an unordered list of sinusoidal peaks 
                 - an ordered list of partials
                 - synthesised audio samples
                 - residual samples
                 - synthesised residual samples"""
    
    def __init__(self):
        self._size = 512
        self._max_partials = 100
        self.audio = None
        self.peaks = None
        self.partials = None
        self.synth = None
        self.residual = None
        self.synth_residual = None


class PeakDetection(object):
    "Detect spectral peaks"
    
    def __init__(self):
        self._sampling_rate = 44100
        self._frame_size = 2048
        self._static_frame_size = True
        self._hop_size = 512
        self._max_peaks = 100
        self._window_type = "hamming"
        self._window_size = 2048
        self._min_peak_separation = 1.0 # in Hz
        self.frames = []
        
    # properties
    sampling_rate = property(lambda self: self.get_sampling_rate(),
                             lambda self, x: self.set_sampling_rate(x))
    frame_size = property(lambda self: self.get_frame_size(),
                          lambda self, x: self.set_frame_size(x))
    hop_size = property(lambda self: self.get_hop_size(),
                        lambda self, x: self.set_hop_size(x))
    max_peaks = property(lambda self: self.get_max_peaks(),
                         lambda self, x: self.set_max_peaks(x))
    window_type = property(lambda self: self.get_window_type(),
                           lambda self, x: self.set_window_type(x))
    window_size = property(lambda self: self.get_window_size(),
                           lambda self, x: self.set_window_size(x))
     
    def get_sampling_rate(self):
        return self._sampling_rate
    
    def set_sampling_rate(self, sampling_rate):
        self._sampling_rate = sampling_rate
        
    def get_frame_size(self):
        return self._frame_size
    
    def set_frame_size(self, frame_size):
        self._frame_size = frame_size
        
    def get_hop_size(self):
        return self._hop_size
    
    def set_hop_size(self, hop_size):
        self._hop_size = hop_size
        
    def get_max_peaks(self):
        return self._max_peaks
    
    def set_max_peaks(self, max_peaks):
        self._max_peaks = max_peaks
        
    def get_window_type(self):
        return self._window_type
        
    def set_window_type(self, window_type):
        self._window_type = window_type
        
    def get_window_size(self):
        return self._window_size
        
    def set_window_size(self, window_size):
        self._window_size = window_size
        
    def get_next_frame_size(self):
        return self._frame_size

    def find_peaks_in_frame(self, frame):
        "Find and return all spectral peaks in a given frame of audio"
        peaks = []
        return peaks
        
    def find_peaks(self, audio):
        """Find and return all spectral peaks in a given audio signal.
        If the signal contains more than 1 frame worth of audio, it will be broken
        up into separate frames, with a list of peaks returned for each frame."""
        self.frames = []
        pos = 0
        while pos < len(audio):
            # get the next frame size
            if not self._static_frame_size:
                self.frame_size = self.get_next_frame_size()
            # get the next frame
            frame = Frame()
            frame.size = self.frame_size
            frame.audio = audio[pos:pos+self.frame_size]
            # pad if necessary
            if len(frame.audio) < self.frame_size:
                frame.audio = np.hstack((frame.audio, 
                                         simpl.zeros(self.frame_size - len(frame.audio))))
            # find peaks
            frame.peaks = self.find_peaks_in_frame(frame)
            self.frames.append(frame)
            pos += self.hop_size
        return self.frames
        

class PartialTracking(object):
    "Link spectral peaks from consecutive frames to form partials"
    def __init__(self):
        self._sampling_rate = 44100
        self._max_partials = 100
        self._min_partial_length = 0
        self._max_gap = 2
        self.frames = []
        
    # properties
    sampling_rate = property(lambda self: self.get_sampling_rate(),
                             lambda self, x: self.set_sampling_rate(x))
    max_partials = property(lambda self: self.get_max_partials(),
                            lambda self, x: self.set_max_partials(x))
    min_partial_length = property(lambda self: self.get_min_partial_length(),
                                  lambda self, x: self.set_min_partial_length(x))
    max_gap = property(lambda self: self.get_max_gap(),
                       lambda self, x: self.set_max_gap(x))
        
    def get_sampling_rate(self):
        return self._sampling_rate
    
    def set_sampling_rate(self, sampling_rate):
        self._sampling_rate = sampling_rate
        
    def get_max_partials(self):
        return self._max_partials
    
    def set_max_partials(self, num_partials):
        self._max_partials = num_partials
        
    def get_min_partial_length(self):
        return self._min_partial_length
    
    def set_min_partial_length(self, length):
        self._min_partial_length = length
        
    def get_max_gap(self):
        return self._max_gap
    
    def set_max_gap(self, gap):
        self._max_gap = gap
        
    def update_partials(self, frame):
        "Streamable (real-time) partial-tracking."
        peaks = [None for i in range(self.max_partials)]
        return peaks
        
    def find_partials(self, frames):
        """Find partials from the sinusoidal peaks in a list of Frames"""
        self.frames = []
        for frame in frames:
            frame.partials = self.update_partials(frame)
            self.frames.append(frame)
        return self.frames

    
class Synthesis(object):
    "Synthesise audio from spectral analysis data"
    def __init__(self):
        self._frame_size = 512
        self._hop_size = 512
        self._max_partials = 100
        self._sampling_rate = 44100
        
    # properties
    frame_size = property(lambda self: self.get_frame_size(),
                          lambda self, x: self.set_frame_size(x))
    hop_size = property(lambda self: self.get_hop_size(),
                        lambda self, x: self.set_hop_size(x))
    max_partials = property(lambda self: self.get_max_partials(),
                            lambda self, x: self.set_max_partials(x))
    max_partials = property(lambda self: self.get_max_partials(),
                            lambda self, x: self.set_max_partials(x))
    sampling_rate = property(lambda self: self.get_sampling_rate(),
                             lambda self, x: self.set_sampling_rate(x))
        
    def get_frame_size(self):
        return self._frame_size
    
    def set_frame_size(self, frame_size):
        self._frame_size = frame_size

    def get_hop_size(self):
        return self._hop_size
    
    def set_hop_size(self, hop_size):
        self._hop_size = hop_size
        
    def get_max_partials(self):
        return self._max_partials
    
    def set_max_partials(self, num_partials):
        self._max_partials = num_partials
        
    def get_sampling_rate(self):
        return self._sampling_rate
    
    def set_sampling_rate(self, sampling_rate):
        self._sampling_rate = sampling_rate

    def synth_frame(self, frame):
        "Synthesises a frame of audio, given a list of peaks from tracks"
        raise Exception("NotYetImplemented")
        
    def synth(self, frames):
        "Synthesise audio from the given partials"
        audio_out = simpl.array([])
        for frame in frames:
            audio_out = np.hstack((audio_out, self.synth_frame(frame)))
        return audio_out
    
    
class Residual(object):
    "Calculate a residual signal"
    
    def __init__(self):
        self._hop_size = 512
        self._frame_size = 512

    def residual_frame(self, synth, original):
        "Computes the residual signal for a frame of audio"
        raise Exception("NotYetImplemented")
        
    def find_residual(self, synth, original):
        "Calculate and return the residual signal"
        # pad the signals if necessary
        if len(synth) % self._hop_size != 0:
            synth = np.hstack((synth, np.zeros(self._hop_size - (len(synth) % self._hop_size))))
        if len(original) % self._hop_size != 0:
            original = np.hstack((original, np.zeros(self._hop_size - (len(original) % self._hop_size))))

        num_frames = len(original) / self._hop_size
        residual = simpl.array([])
        sample_offset = 0

        for i in range(num_frames):
            synth_frame = synth[sample_offset:sample_offset+self._hop_size]
            original_frame = original[sample_offset:sample_offset+self._hop_size]
            residual = np.hstack((residual, 
                                  self.residual_frame(synth_frame, original_frame)))
            sample_offset += self._hop_size
        return residual

    def synth_frame(self, synth, original):
        "Calculate and return one frame of the synthesised residual signal"
        raise Exception("NotYetImplemented")
    
    def synth(self, synth, original):
        "Calculate and return a synthesised residual signal"
        # pad the signals if necessary
        if len(synth) % self._hop_size != 0:
            synth = np.hstack((synth, np.zeros(self._hop_size - (len(synth) % self._hop_size))))
        if len(original) % self._hop_size != 0:
            original = np.hstack((original, np.zeros(self._hop_size - (len(original) % self._hop_size))))

        num_frames = len(original) / self._hop_size
        residual = simpl.array([])
        sample_offset = 0

        for i in range(num_frames):
            synth_frame = synth[sample_offset:sample_offset+self._hop_size]
            original_frame = original[sample_offset:sample_offset+self._hop_size]
            residual = np.hstack((residual, 
                                  self.synth_frame(synth_frame, original_frame)))
            sample_offset += self._hop_size
        return residual

