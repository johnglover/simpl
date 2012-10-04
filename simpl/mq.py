import simpl
import numpy as np
import operator as op


def best_match(f, candidates):
    best_diff = 22050.0
    pos = 0
    for i, c in enumerate(candidates):
        if abs(f - c) < best_diff:
            best_diff = abs(f - c)
            pos = i
    return pos


def twm(peaks, f_min=0.0, f_max=3000.0, f_step=20.0):
    # twm parameters
    p = 0.5
    q = 1.4
    r = 0.5
    rho = 0.33
    N = 8
    Err = {}

    max_amp = max([x.amplitude for x in peaks])

    # remove all peaks with amplitude of less than 10% of max
    # note: this is not in the TWM paper, found that it improved
    # accuracy however
    peaks = [x for x in peaks if x.amplitude >= (max_amp * 0.1)]

    # get the max frequency of the remaining peaks
    max_freq = max([x.frequency for x in peaks])
    if max_freq < f_max:
        f_max = max_freq

    f_current = f_min
    while f_current < f_max:
        Err_pm = 0.0
        Err_mp = 0.0
        harmonics = np.arange(f_current, f_max, f_current)
        if len(harmonics) > N:
            harmonics = harmonics[0:N]

        # calculate mismatch between predicted and actual peaks
        for h in harmonics:
            k = best_match(f_current, [x.frequency for x in peaks])
            f = peaks[k].frequency
            a = peaks[k].amplitude
            Err_pm += abs(h - f) * (h ** -p) + (a / max_amp) * ((q * (abs(h - f)) * (h ** -p) - r))

        # calculate the mismatch between actual and predicted peaks
        for x in peaks:
            k = best_match(x.frequency, harmonics)
            f = harmonics[k]
            xf = x.frequency
            a = x.amplitude
            Err_mp += abs(xf - f) * (xf ** -p) + (a / max_amp) * ((q * (abs(xf - f)) * (xf ** -p) - r))

        # calculate the total error for f_current as a fundamental frequency
        Err[f_current] = (Err_pm / len(harmonics)) + (rho * Err_mp / len(peaks))
        f_current += f_step

    # return the value with the minimum total error
    return min(Err.iteritems(), key=op.itemgetter(1))[0]


class MQPeakDetection(simpl.PeakDetection):
    """
    Peak detection, based on the McAulay and Quatieri (MQ) algorithm.

    A peak is defined as the point in the spectrum where the slope changes from
    position to negative. Hamming window is used, window size must be
    (at least) 2.5 times the average pitch. During voiced sections of speech,
    the window size is updated every 0.25 secs, to the average pitch.
    During unvoiced sections, the window size is fixed at the value of the
    last voiced frame. Once the width is specified, the Hamming window is
    computed and normalised and the STFT is calculated.
    """
    def __init__(self):
        simpl.PeakDetection.__init__(self)
        self._frame_size = super(MQPeakDetection, self).frame_size
        self._create_analysis_window()
        self._fundamental = float(self.sampling_rate) / self._frame_size
        self.static_frame_size = False
        self._current_peaks = []
        self._freq_estimates = []
        # no. frames to use to estimate the average pitch (1/4 second window)
        self._avg_freq_frames = int(
            0.25 * self.sampling_rate / self._frame_size
        )

    @property
    def frame_size(self):
        return self._frame_size

    @frame_size.setter
    def frame_size(self, new_frame_size):
        self._frame_size = new_frame_size
        self._fundamental = float(self.sampling_rate) / self._frame_size
        self._create_analysis_window()

    def _create_analysis_window(self):
        "Creates the analysis window, a normalised hamming window"
        self._window = np.hamming(self._frame_size)
        self._window /= np.sum(self._window)

    def next_frame_size(self):
        if not len(self._current_peaks):
            return self._frame_size

        # frame size must be at least 2.5 times the average pitch period,
        # where the average is taken over 1/4 second.
        #
        # TODO: average should not include frames corresponding to unvoiced
        # speech, ie noisy frames
        self._freq_estimates.append(
            twm(self._current_peaks, f_min=self._fundamental,
                f_step=self._fundamental)
        )
        if len(self._freq_estimates) > self._avg_freq_frames:
            self._freq_estimates.pop(0)

        avg_freq = sum(self._freq_estimates) / len(self._freq_estimates)
        pitch_period = float(self.sampling_rate) / avg_freq

        if self._frame_size < (2.5 * pitch_period):
            return int(2.5 * pitch_period)
        else:
            return self._frame_size

    def find_peaks_in_frame(self, frame):
        """
        Selects the highest peaks from the given spectral frame, up to a
        maximum of self._max_peaks peaks.
        """
        self._current_peaks = []

        if frame.max_peaks != self.max_peaks:
            frame.max_peaks = self.max_peaks

        f = np.fft.rfft(frame.audio * self._window)
        spectrum = abs(f)

        # find all peaks in the spectrum
        prev_mag = np.abs(spectrum[0])
        current_mag = np.abs(spectrum[1])
        next_mag = 0.0

        for bin in range(2, len(spectrum) - 1):
            next_mag = np.abs(spectrum[bin])
            if (current_mag > prev_mag and
                current_mag > next_mag):
                p = simpl.Peak()
                p.amplitude = current_mag
                p.frequency = (bin - 1) * self._fundamental
                p.phase = np.angle(f[bin - 1])
                self._current_peaks.append(p)
            prev_mag = current_mag
            current_mag = next_mag

        # sort peaks, largest amplitude first, and up to a max
        # of self.num_peaks peaks
        self._current_peaks.sort(cmp=simpl.compare_peak_amps)
        self._current_peaks.reverse()
        if len(self._current_peaks) > self.max_peaks:
            self._current_peaks = self._current_peaks[0:self.max_peaks]

        # put back into ascending frequency order
        self._current_peaks.sort(cmp=simpl.compare_peak_freqs)
        frame.peaks = list(self._current_peaks)
        return self._current_peaks


class MQPartialTracking(simpl.PartialTracking):
    "Partial tracking using the McAulay and Quatieri (MQ) algorithm"
    def __init__(self):
        simpl.PartialTracking.__init__(self)
        self._matching_interval = 100  # peak matching interval, in Hz
        self._current_frame = None  # current frame in peak tracking

    def _find_closest_match(self, peak, frame_peaks, matched_peaks):
        """
        Find a candidate match for peak in frame if one exists.
        This is the closest (in frequency) match that is within
        self._matching_interval.
        """
        free_peaks = [p for p in frame_peaks if not p in matched_peaks]
        free_peaks = [p for p in free_peaks if p.amplitude > 0]
        distances = [abs(peak.frequency - p.frequency) for p in free_peaks]
        if len(distances):
            min_distance_position = min(xrange(len(distances)),
                                        key=distances.__getitem__)
            if min(distances) < self._matching_interval:
                return free_peaks[min_distance_position]
        return None

    def _get_free_peak_below(self, peak, frame_peaks, matched_peaks):
        """
        Returns the closest unmatched peak in frame_peaks with a frequency
        less than peak.frequency.
        """
        freqs = [p.frequency for p in matched_peaks if p]
        for peak_number, p in enumerate(frame_peaks):
            if p.frequency == peak.frequency:
                # go back through lower peaks (in order) and
                # return the first unmatched
                current_peak = peak_number - 1
                while current_peak >= 0:
                    if not frame_peaks[current_peak].frequency in freqs:
                        return frame_peaks[current_peak]
                    current_peak -= 1
                return None
        return None

    def _kill_partial(self, partials, prev_peak):
        """
        When a partial dies it is matched to itself in the next frame,
        with 0 amplitude.
        """
        for peak_number, peak in enumerate(self._current_frame.partials):
            if peak.frequency == prev_peak.frequency:
                if peak.amplitude == 0:
                    partials[peak_number] = None
                else:
                    p = simpl.Peak()
                    p.frequency = peak.frequency
                    partials[peak_number] = p

    def _extend_partial(self, partials, prev_peak, next_peak):
        """
        Sets next_peak to be the next sinusoidal peak in the partial
        that currently ends with prev_peak.
        """
        for peak_number, peak in enumerate(self._current_frame.partials):
            if peak.frequency == prev_peak.frequency:
                partials[peak_number] = next_peak

    def update_partials(self, frame):
        """
        Streamable (real-time) MQ partial-tracking.

        1. If there is no peak within the matching interval, the track dies.
        If there is at least one peak within the matching interval, the closest
        match is declared a candidate match.

        2. If there is a candidate match from step 1, and it is not a closer
        match to any of the remaining unmatched frequencies, it is declared a
        definitive match.  If the candidate match has a closer unmatched peak
        in the previous frame, it is not selected. Instead, the closest lower
        frequency peak to the candidate is checked. If it is within the
        matching interval, it is selected as a definitive match. If not, the
        track dies. In any case, step 1 is repeated on the next unmatched peak.

        3. Once all peaks from the current frame have been matched, there may
        still be peaks remaining in the next frame. A new peak is created in
        the current frame at the same frequency and with 0 amplitude, and a
        match is made.
        """
        if not frame.max_partials == self.max_partials:
            frame.max_partials = self.max_partials

        partials = [None for i in range(self.max_partials)]

        # MQ algorithm needs 2 frames of data, so create new partials and
        # return if this is the first frame
        if not self._current_frame:
            self._current_frame = frame
            # if more peaks than paritals, select the max_partials largest
            # amplitude peaks in frame
            if len(frame.peaks) > self.max_partials:
                frame.peaks.sort(cmp=simpl.compare_peak_amps)
                frame.peaks.reverse()
                partials = frame.peaks[0:self.max_partials]
            # if not, save all peaks as new partials, and add a few zero
            # peaks if necessary
            else:
                partials = frame.peaks
                for i in range(len(frame.peaks), self.max_partials):
                    partials.append(simpl.Peak())
            frame.partials = partials
            return partials

        for peak in self._current_frame.partials:
            match = self._find_closest_match(peak, frame.peaks, partials)
            if match:
                # is this match closer to any of the other unmatched
                # peaks in frame?
                closest_to_candidate = self._find_closest_match(
                    match, self._current_frame.partials, []
                )
                if not closest_to_candidate == peak:
                    # see if the closest peak with lower frequency to
                    # the candidate is within the matching interval
                    lower_peak = self._get_free_peak_below(
                        match, frame.peaks, partials
                    )
                    if lower_peak:
                        if abs(lower_peak.frequency - peak.frequency) < \
                            self._matching_interval:
                            # this is the definitive match
                            self._extend_partial(partials, peak, lower_peak)
                        else:
                            self._kill_partial(partials, peak)
                    else:
                        self._kill_partial(partials, peak)
                # if not, it is a definitive match
                else:
                    self._extend_partial(partials, peak, match)
            else:  # no match
                self._kill_partial(partials, peak)

        # now that all peaks in the current frame have been matched,
        # look for any unmatched peaks in the next frame
        for p in frame.peaks:
            # if not p.previous_peak:
            if not p in partials:
                # look for the first free partial spot in partials
                free_partial = -1
                for i in range(self.max_partials):
                    if not partials[i]:
                        free_partial = i
                        break
                if not free_partial == -1:
                    # create a new track by adding a peak in the current frame
                    partials[free_partial] = p

        # add zero peaks for any remaining free partials
        for i in range(self.max_partials):
            if not partials[i]:
                partials[i] = simpl.Peak()

        self._current_frame = frame
        frame.partials = partials
        return frame


class MQSynthesis(simpl.Synthesis):
    def __init__(self):
        simpl.Synthesis.__init__(self)
        self._max_partials = super(MQSynthesis, self).max_partials
        self._previous_partials = [
            simpl.Peak() for i in range(self._max_partials)
        ]

    @property
    def max_partials(self):
        return self._max_partials

    @max_partials.setter
    def max_partials(self, new_max_partials):
        self._max_partials = new_max_partials
        self._previous_partials = [
            simpl.Peak() for i in range(self._max_partials)
        ]

    def hz_to_radians(self, frequency):
        if not frequency:
            return 0.0
        else:
            return (frequency * 2.0 * np.pi) / self.sampling_rate

    def synth_frame(self, frame):
        output = np.zeros(self.hop_size, dtype=simpl.dtype)
        size = self.hop_size

        for n, p in enumerate(frame.partials):
            # get values for last amplitude, frequency and phase
            # these are the initial values of the instantaneous
            # amplitude/frequency/phase
            current_freq = self.hz_to_radians(p.frequency)
            prev_amp = self._previous_partials[n].amplitude
            if prev_amp == 0:
                prev_freq = current_freq
                prev_phase = p.phase - (current_freq * size)
                while prev_phase >= np.pi:
                    prev_phase -= 2.0 * np.pi
                while prev_phase < -np.pi:
                    prev_phase += 2.0 * np.pi
            else:
                prev_freq = self.hz_to_radians(
                    self._previous_partials[n].frequency
                )
                prev_phase = self._previous_partials[n].phase

            # amplitudes are linearly interpolated between frames
            inst_amp = prev_amp
            amp_inc = (p.amplitude - prev_amp) / size

            # freqs/phases are calculated by cubic interpolation
            freq_diff = current_freq - prev_freq
            x = ((prev_phase + (prev_freq * size) - p.phase) +
                 (freq_diff * (size / 2.0)))
            x /= (2.0 * np.pi)
            m = int(np.round(x))
            phase_diff = p.phase - prev_phase - (prev_freq * size) + \
                         (2.0 * np.pi * m)
            alpha = ((3.0 / (size ** 2)) * phase_diff) - (freq_diff / size)
            beta = ((-2.0 / (size ** 3)) * phase_diff) + \
                   (freq_diff / (size ** 2))

            # calculate output samples
            for i in range(size):
                inst_amp += amp_inc
                inst_phase = prev_phase + (prev_freq * i) + \
                             (alpha * (i ** 2)) + (beta * (i ** 3))
                output[i] += (2.0 * inst_amp) * np.cos(inst_phase)

            # update previous partials list
            self._previous_partials[n] = p

        frame.synth = output
        return output
