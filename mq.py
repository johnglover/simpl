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

class MQPeakDetection(simpl.PeakDetection):
    """Peak detection, based on the McAulay and Quatieri (MQ) algorithm.
    
    A peak is defined as the point in the spectrum where the slope changes from
    position to negative. Hamming window is used, window size must be (at least) 
    2.5 times the average pitch. During voiced sections of speech, the window size
    is updated every 0.25 secs, to the average pitch. During unvoiced sections,
    the window size is fixed at the value of the last voiced frame. Once the width
    is specified, the Hamming window is computed and normalised and the STFT is
    calculated."""
    def __init__(self):
        simpl.PeakDetection.__init__(self)
        self._window = simpl.zeros(self._window_size)
        self._create_analysis_window()
        self._fundamental = float(self._sampling_rate) / self._window_size
        
    def set_window_size(self, window_size):
        self._window_size = window_size
        self._fundamental = float(self._sampling_rate) / window_size
        
    def _create_analysis_window(self):
        "Creates the analysis window, a normalised hamming window"
        self._window = np.hamming(self._window_size)
        s = 0
        for i in range(self._window_size):
            s += self._window[i]
        self._window /= s
        
    def find_peaks_in_frame(self, frame):
        """Selects the highest peaks from the given spectral frame, up to a maximum of 
        self._max_peaks peaks."""
        current_peaks = []        
        # fft of frame
        f = np.fft.rfft(frame * self._window)
        spectrum = abs(f)
        # find all peaks in the spectrum
        prev_mag = np.abs(spectrum[0])
        current_mag = np.abs(spectrum[1])
        next_mag = 0.0
        for bin in range(2, len(spectrum)-1):
            next_mag = np.abs(spectrum[bin])
            if (current_mag > prev_mag and 
                current_mag > next_mag):
                p = simpl.Peak()
                p.amplitude = current_mag
                p.frequency = (bin - 1) * self._fundamental
                p.phase = np.angle(spectrum[bin-1])
                current_peaks.append(p)
            prev_mag = current_mag
            current_mag = next_mag
        # sort peaks, largest amplitude first, and up to a max of self.num_peaks peaks        
        current_peaks.sort(cmp=simpl.compare_peak_amps)   
        current_peaks.reverse()
        if len(current_peaks) > self._max_peaks:
            current_peaks = current_peaks[0:self._max_peaks]
        # put back into ascending frequency order
        current_peaks.sort(cmp=simpl.compare_peak_freqs)
        return current_peaks


class MQPartialTracking(simpl.PartialTracking):
    "Partial tracking, based on the McAulay and Quatieri (MQ) algorithm"    
    def __init__(self):
        simpl.PartialTracking.__init__(self)
        self._matching_interval = 100  # peak matching interval, in Hz
        self._current_frame = None  # current frame in the peak tracking algorithm

    def _find_closest_match(self, peak, frame, direction='backwards'):
        """Find a candidate match for peak in frame if one exists. This is the closest
        (in frequency) match that is within self._matching_interval."""
        free_peaks = []
        for p in frame:
            if p.is_free(direction):
                free_peaks.append(p)
        distances = [abs(peak.frequency - p.frequency) for p in free_peaks]
        if len(distances):
            min_distance_position = min(xrange(len(distances)), key=distances.__getitem__)
            if min(distances) < self._matching_interval:
                return free_peaks[min_distance_position]
        return None
        
    def _get_free_peak_below(self, peak, frame, direction='backwards'):
        """Returns the closest unmatched peak in frame with a frequency less than peak.frequency."""
        # find peak in frame
        for peak_number, p in enumerate(frame):
            if p == peak:
                # go back through lower peaks (in order) and return the first unmatched
                current_peak = peak_number - 1
                while current_peak >= 0:
                    if frame[current_peak].is_free(direction):
                        return frame[current_peak]
                    current_peak -= 1
                return None
        return None
    
    def _kill_track(self, peak, next_frame):
        """When a track dies it is matched to itself in the next frame, with 0 amplitude."""
        if peak.is_free():  # this may be a 0 amp (dead) peak from a previous tracking step
            next_peak = simpl.Peak()
            next_peak.amplitude = 0
            next_peak.frequency = peak.frequency
            self.get_partial(peak.partial_id).add_peak(next_peak)
        
    def update_partials(self, frame, frame_number):
        """Streamable (real-time) MQ peak-tracking.
        
        1. If there is no peak within the matching interval, the track dies.
        If there is at least one peak within the matching interval, the closest match
        is declared a candidate match.

        2. If there is a candidate match from step 1, and it is not a closer match to
        any of the remaining unmatched frequencies, it is declared a definitive match.
        If the candidate match has a closer unmatched peak in the previous frame, it is
        not selected. Instead, the closest lower frequency peak to the candidate is
        checked. If it is within the matching interval, it is selected as a definitive
        match. If not, the track dies.
        In any case, step 1 is repeated on the next unmatched peak.

        3. Once all peaks from the current frame have been matched, there may still be
        peaks remaining in the next frame. A new peak is created in the current frame
        at the same frequency and with 0 amplitude, and a match is made."""
        frame_partials = []
        # MQ algorithm needs 2 frames of data, so return if this is the first frame
        if not self._current_frame:
            self._current_frame = frame
            return frame_partials
                
        for peak in self._current_frame:
            if peak.is_start_of_partial():
                partial = simpl.Partial()
                partial.starting_frame = frame_number-1
                partial.add_peak(peak)
                self.partials.append(partial)
            match = self._find_closest_match(peak, frame)
            if match:
                # is this match closer to any of the other unmatched peaks in frame?
                closest_to_candidate = self._find_closest_match(match, self._current_frame, 'forwards')
                if not closest_to_candidate == peak:
                    # see if the closest peak with lower frequency to the candidate is within
                    # the matching interval
                    lower_peak = self._get_free_peak_below(match, frame)
                    if lower_peak:
                        if abs(lower_peak.frequency - peak.frequency) < self._matching_interval:
                            # this is the definitive match
                            self.get_partial(peak.partial_id).add_peak(lower_peak)
                        else:
                            self._kill_track(peak, frame)
                    else:
                        self._kill_track(peak, frame)
                # if not, it is a definitive match
                else:
                    self.get_partial(peak.partial_id).add_peak(match)
            else:  # no match
                self._kill_track(peak, frame)
                
        # now that all peaks in the current frame have been matched, look for any 
        # unmatched peaks in the next frame
        for p in frame:
            if not p.previous_peak:
                # create a new track by adding a peak in the current frame, matched to p,
                # with amplitude 0
                partial = simpl.Partial()
                partial.starting_frame = frame_number-1
                new_peak = simpl.Peak()
                new_peak.amplitude = 0
                new_peak.frequency = p.frequency
                partial.add_peak(new_peak)
                partial.add_peak(p)
                self.partials.append(partial)
        self._current_frame = frame        
        return frame_partials

