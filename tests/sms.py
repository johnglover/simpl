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
import pysms
import numpy as np
from scipy.io.wavfile import read
import unittest

class TestSimplSMS(unittest.TestCase):
    FLOAT_PRECISION = 3 # number of decimal places to check for accuracy
    input_file = 'audio/flute.wav'
    frame_size = 2048
    hop_size = 512
    num_frames = 9 
    num_samples = frame_size + ((num_frames - 1) * hop_size)
    max_peaks = 10
    max_partials = 3

    def get_audio(self):
        audio_data = read(self.input_file)
        audio = simpl.asarray(audio_data[1]) / 32768.0
        sampling_rate = audio_data[0]
        return audio[0:self.num_samples], sampling_rate

    def pysms_analysis_params(self, sampling_rate):
        analysis_params = pysms.SMS_AnalParams()
        pysms.sms_initAnalParams(analysis_params)
        analysis_params.iSamplingRate = sampling_rate
        analysis_params.iFrameRate = sampling_rate / self.hop_size
        analysis_params.iWindowType = pysms.SMS_WIN_HAMMING
        analysis_params.fDefaultFundamental = 100
        analysis_params.fHighestFreq = 20000
        analysis_params.iFormat = pysms.SMS_FORMAT_HP
        analysis_params.nTracks = self.max_peaks
        analysis_params.peakParams.iMaxPeaks = self.max_peaks
        analysis_params.nGuides = self.max_peaks
        analysis_params.iMaxDelayFrames = 4
        analysis_params.analDelay = 0
        analysis_params.minGoodFrames = 1
        analysis_params.iCleanTracks = 0
        analysis_params.iStochasticType = pysms.SMS_STOC_NONE
        return analysis_params

    def simplsms_analysis_params(self, sampling_rate):
        analysis_params = simplsms.SMS_AnalParams()
        simplsms.sms_initAnalParams(analysis_params)
        analysis_params.iSamplingRate = sampling_rate
        analysis_params.iFrameRate = sampling_rate / self.hop_size
        analysis_params.iWindowType = simplsms.SMS_WIN_HAMMING
        analysis_params.fDefaultFundamental = 100
        analysis_params.fHighestFreq = 20000
        analysis_params.iFormat = simplsms.SMS_FORMAT_HP
        analysis_params.nTracks = self.max_peaks
        analysis_params.maxPeaks = self.max_peaks
        analysis_params.nGuides = self.max_peaks
        analysis_params.iMaxDelayFrames = 4
        analysis_params.analDelay = 0
        analysis_params.minGoodFrames = 1
        analysis_params.iCleanTracks = 0
        analysis_params.iStochasticType = simplsms.SMS_STOC_NONE
        return analysis_params

    def test_size_next_read(self):
        """test_size_next_read
        Make sure pysms PeakDetection is calculating 
        the correct value for the size of the next frame."""
        audio, sampling_rate = self.get_audio()

        pysms.sms_init()
        snd_header = pysms.SMS_SndHeader()
        # Try to open the input file to fill snd_header
        if(pysms.sms_openSF(self.input_file, snd_header)):
            raise NameError("error opening sound file: " + pysms.sms_errorString())
        analysis_params = self.pysms_analysis_params(sampling_rate)
        analysis_params.iMaxDelayFrames = self.num_frames + 1
        if pysms.sms_initAnalysis(analysis_params, snd_header) != 0:
            raise Exception("Error allocating memory for analysis_params")
        analysis_params.nFrames = self.num_frames
        sms_header = pysms.SMS_Header()
        pysms.sms_fillHeader(sms_header, analysis_params, "pysms")

        sample_offset = 0
        pysms_size_new_data = 0
        current_frame = 0
        sms_next_read_sizes = []

        while current_frame < self.num_frames:
            sms_next_read_sizes.append(analysis_params.sizeNextRead)
            sample_offset += pysms_size_new_data
            if((sample_offset + analysis_params.sizeNextRead) < self.num_samples):
                pysms_size_new_data = analysis_params.sizeNextRead
            else:
                pysms_size_new_data = self.num_samples - sample_offset
            # convert frame to floats for libsms
            frame = audio[sample_offset:sample_offset + pysms_size_new_data]
            frame = np.array(frame, dtype=np.float32)
            analysis_data = pysms.SMS_Data()
            pysms.sms_allocFrameH(sms_header, analysis_data)
            status = pysms.sms_analyze(frame, analysis_data, analysis_params)  
            # as the no. of frames of delay is > num_frames, sms_analyze should
            # never get around to performing partial tracking, and so the return
            # value should be 0
            self.assertEquals(status, 0)
            pysms.sms_freeFrame(analysis_data)
            current_frame += 1

        pysms.sms_freeAnalysis(analysis_params)
        pysms.sms_closeSF()
        pysms.sms_free()

        pd = simpl.SMSPeakDetection()
        pd.hop_size = self.hop_size
        current_frame = 0
        sample_offset = 0

        while current_frame < self.num_frames:
            pd.frame_size = pd.get_next_frame_size()
            self.assertEquals(sms_next_read_sizes[current_frame], pd.frame_size)
            pd.find_peaks_in_frame(audio[sample_offset:sample_offset + pd.frame_size])
            sample_offset += pd.frame_size
            current_frame += 1

    def test_peak_detection(self): 
        """test_peak_detection
        Compare simplsms Peaks with SMS peaks. Exact peak
        information cannot be retrieved using libsms. Basic peak detection
        is performed by sms_detectPeaks, but this is called multiple times
        with different frame sizes by sms_analyze. This peak data cannot
        be returned from sms_analyze without modifying it, so here
        we compare the peaks to a slightly modified version of sms_analyze 
        from simplsms. The peak values should be the same as those found by 
        the simplsms find_peaks function. Analyses have to be performed
        separately due to libsms implementation issues."""
        audio, sampling_rate = self.get_audio()

        simplsms.sms_init()
        analysis_params = self.simplsms_analysis_params(sampling_rate)
        analysis_params.iMaxDelayFrames = self.num_frames + 1
        if simplsms.sms_initAnalysis(analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")
        analysis_params.nFrames = self.num_frames
        sms_header = simplsms.SMS_Header()
        simplsms.sms_fillHeader(sms_header, analysis_params, "simplsms")

        sample_offset = 0
        size_new_data = 0
        current_frame = 0
        sms_peaks = []

        while current_frame < self.num_frames:
            sample_offset += size_new_data
            size_new_data = analysis_params.sizeNextRead
            frame = audio[sample_offset:sample_offset + size_new_data]
            analysis_data = simplsms.SMS_Data()
            simplsms.sms_allocFrameH(sms_header, analysis_data)
            status = simplsms.sms_analyze(frame, analysis_data, analysis_params)  
            # as the no. of frames of delay is > num_frames, sms_analyze should
            # never get around to performing partial tracking, and so the return
            # value should be 0
            self.assertEquals(status, 0)
            num_peaks = analysis_data.nTracks
            frame_peaks = []
            simplsms_freqs = simpl.zeros(num_peaks)
            simplsms_amps = simpl.zeros(num_peaks)
            simplsms_phases = simpl.zeros(num_peaks)
            analysis_data.getSinFreq(simplsms_freqs)
            analysis_data.getSinAmp(simplsms_amps)
            analysis_data.getSinPhase(simplsms_phases)
            for i in range(num_peaks):
                if simplsms_amps[i]:
                    p = simpl.Peak()
                    # convert amplitude back to linear
                    p.amplitude = 10**(simplsms_amps[i]/20.0)
                    p.frequency = simplsms_freqs[i]
                    p.phase = simplsms_phases[i]
                    frame_peaks.append(p)
            sms_peaks.append(frame_peaks)
            pysms.sms_freeFrame(analysis_data)
            current_frame += 1

        simplsms.sms_freeAnalysis(analysis_params)
        simplsms.sms_free()

        # get simpl peaks
        pd = simpl.SMSPeakDetection()
        pd.hop_size = self.hop_size 
        pd.max_peaks = self.max_peaks
        current_frame = 0
        sample_offset = 0
        simpl_peaks = []

        while current_frame < self.num_frames:
            pd.frame_size = pd.get_next_frame_size()
            simpl_peaks.append(
                    pd.find_peaks_in_frame(audio[sample_offset:sample_offset + pd.frame_size]))
            sample_offset += pd.frame_size
            current_frame += 1

        # make sure we have the same number of frames
        self.assertEquals(len(sms_peaks), len(simpl_peaks))

        # compare data for each frame
        for frame_number in range(len(sms_peaks)):
            sms_frame = sms_peaks[frame_number]
            simpl_frame = simpl_peaks[frame_number]
            # make sure we have the same number of peaks in each frame
            self.assertEquals(len(sms_frame), len(simpl_frame))
            # check peak values
            for peak_number in range(len(sms_frame)):
                #print frame_number, peak_number
                sms_peak = sms_frame[peak_number]
                simpl_peak = simpl_frame[peak_number]
                self.assertAlmostEquals(sms_peak.amplitude, simpl_peak.amplitude,
                                        places=self.FLOAT_PRECISION)
                self.assertAlmostEquals(sms_peak.frequency, simpl_peak.frequency,
                                        places=self.FLOAT_PRECISION)
                self.assertAlmostEquals(sms_peak.phase, simpl_peak.phase,
                                        places=self.FLOAT_PRECISION)  

    def test_sms_analyze(self):
        """test_sms_analyzebt43lztar
        Make sure that the simplsms.sms_analyze function does the same thing
        as the sms_analyze function from libsms."""
        audio, sampling_rate = self.get_audio()

        pysms.sms_init()
        snd_header = pysms.SMS_SndHeader()
        # Try to open the input file to fill snd_header
        if(pysms.sms_openSF(self.input_file, snd_header)):
            raise NameError("error opening sound file: " + pysms.sms_errorString())
        analysis_params = self.pysms_analysis_params(sampling_rate)
        analysis_params.iMaxDelayFrames = self.num_frames + 1
        analysis_params.analDelay = 0
        analysis_params.minGoodFrames = 1
        if pysms.sms_initAnalysis(analysis_params, snd_header) != 0:
            raise Exception("Error allocating memory for analysis_params")
        analysis_params.nFrames = self.num_frames
        analysis_params.iSizeSound = self.num_samples
        analysis_params.peakParams.iMaxPeaks = self.max_peaks
        sms_header = pysms.SMS_Header()
        pysms.sms_fillHeader(sms_header, analysis_params, "pysms")

        sample_offset = 0
        size_new_data = 0
        current_frame = 0
        sms_partials = []
        live_partials = [None for i in range(self.max_peaks)]
        do_analysis = True

        while do_analysis and (current_frame < self.num_frames):
            sample_offset += size_new_data
            size_new_data = analysis_params.sizeNextRead
            # convert frame to floats for libsms
            frame = audio[sample_offset:sample_offset + size_new_data]
            frame = np.array(frame, dtype=np.float32)
            analysis_data = pysms.SMS_Data()
            pysms.sms_allocFrameH(sms_header, analysis_data)
            status = pysms.sms_analyze(frame, analysis_data, analysis_params)  

            if status == 1:
                num_partials = analysis_data.nTracks
                sms_freqs = np.zeros(num_partials, dtype=np.float32)
                sms_amps = np.zeros(num_partials, dtype=np.float32)
                sms_phases = np.zeros(num_partials, dtype=np.float32)
                analysis_data.getSinFreq(sms_freqs)
                analysis_data.getSinAmp(sms_amps)
                analysis_data.getSinPhase(sms_phases)
                # make partial objects
                for i in range(num_partials):
                    # for each partial, if the mag is > 0, this partial is alive
                    if sms_amps[i] > 0:
                        # create a peak object
                        p = simpl.Peak()
                        p.amplitude = sms_amps[i]
                        p.frequency = sms_freqs[i]
                        p.phase = sms_phases[i]
                        # add this peak to the appropriate partial
                        if not live_partials[i]:
                            live_partials[i] = simpl.Partial()
                            live_partials[i].starting_frame = current_frame
                            sms_partials.append(live_partials[i])
                        live_partials[i].add_peak(p)
                    # if the mag is 0 and this partial was alive, kill it
                else:
                    if live_partials[i]:
                        live_partials[i] = None
            elif status == -1:
                do_analysis = False
            pysms.sms_freeFrame(analysis_data)
            current_frame += 1

        pysms.sms_freeAnalysis(analysis_params)
        pysms.sms_closeSF()
        pysms.sms_free()

        audio, sampling_rate = self.get_audio()
        simplsms.sms_init()
        simpl_analysis_params = self.simplsms_analysis_params(sampling_rate)
        simpl_analysis_params.iMaxDelayFrames = self.num_frames + 1
        if simplsms.sms_initAnalysis(simpl_analysis_params) != 0:
            raise Exception("Error allocating memory for analysis_params")
        simpl_analysis_params.nFrames = self.num_frames
        simpl_analysis_params.iSizeSound = self.num_samples
        simpl_sms_header = simplsms.SMS_Header()
        simplsms.sms_fillHeader(simpl_sms_header, simpl_analysis_params, "simplsms")

        sample_offset = 0
        size_new_data = 0
        current_frame = 0
        simplsms_partials = []
        live_partials = [None for i in range(self.max_peaks)]
        do_analysis = True

        while do_analysis and (current_frame < self.num_frames):
            sample_offset += size_new_data
            size_new_data = simpl_analysis_params.sizeNextRead
            frame = audio[sample_offset:sample_offset + size_new_data]
            analysis_data = simplsms.SMS_Data()
            simplsms.sms_allocFrameH(simpl_sms_header, analysis_data)
            status = simplsms.sms_analyze(frame, analysis_data, simpl_analysis_params)  

            if status == 1:
                num_partials = analysis_data.nTracks
                freqs = simpl.zeros(num_partials)
                amps = simpl.zeros(num_partials)
                phases = simpl.zeros(num_partials)
                analysis_data.getSinFreq(freqs)
                analysis_data.getSinAmp(amps)
                analysis_data.getSinPhase(phases)
                # make partial objects
                for i in range(num_partials):
                    # for each partial, if the mag is > 0, this partial is alive
                    if amps[i] > 0:
                        # create a peak object
                        p = simpl.Peak()
                        p.amplitude = amps[i]
                        p.frequency = freqs[i]
                        p.phase = phases[i]
                        # add this peak to the appropriate partial
                        if not live_partials[i]:
                            live_partials[i] = simpl.Partial()
                            live_partials[i].starting_frame = current_frame
                            simplsms_partials.append(live_partials[i])
                        live_partials[i].add_peak(p)
                    # if the mag is 0 and this partial was alive, kill it
                else:
                    if live_partials[i]:
                        live_partials[i] = None
            elif status == -1:
                do_analysis = False
            simplsms.sms_freeFrame(analysis_data)
            current_frame += 1

        simplsms.sms_freeAnalysis(simpl_analysis_params)
        simplsms.sms_free()

        # make sure both have the same number of partials
        self.assertEquals(len(sms_partials), len(simplsms_partials))

        # make sure each partial is the same
        for i in range(len(sms_partials)):
            self.assertEquals(sms_partials[i].get_length(), simplsms_partials[i].get_length())
            for peak_number in range(sms_partials[i].get_length()):
                self.assertAlmostEquals(sms_partials[i].peaks[peak_number].amplitude,
                                        simplsms_partials[i].peaks[peak_number].amplitude,
                                        places = self.FLOAT_PRECISION)
                self.assertAlmostEquals(sms_partials[i].peaks[peak_number].frequency,
                                        simplsms_partials[i].peaks[peak_number].frequency,
                                        places = self.FLOAT_PRECISION)
                self.assertAlmostEquals(sms_partials[i].peaks[peak_number].phase,
                                        simplsms_partials[i].peaks[peak_number].phase,
                                        places = self.FLOAT_PRECISION)

    def test_partial_tracking(self):
        """test_partial_tracking
        Compare pysms Partials with SMS partials.""" 
        audio, sampling_rate = self.get_audio()

        pysms.sms_init()
        snd_header = pysms.SMS_SndHeader()
        # Try to open the input file to fill snd_header
        if(pysms.sms_openSF(self.input_file, snd_header)):
            raise NameError("error opening sound file: " + pysms.sms_errorString())
        analysis_params = self.pysms_analysis_params(sampling_rate)
        if pysms.sms_initAnalysis(analysis_params, snd_header) != 0:
            raise Exception("Error allocating memory for analysis_params")
        analysis_params.iSizeSound = self.num_samples
        sms_header = pysms.SMS_Header()
        pysms.sms_fillHeader(sms_header, analysis_params, "pysms")

        sample_offset = 0
        size_new_data = 0
        current_frame = 0
        sms_partials = []
        live_partials = [None for i in range(self.max_peaks)]
        do_analysis = True

        while do_analysis and (current_frame < self.num_frames):
            sample_offset += size_new_data
            size_new_data = analysis_params.sizeNextRead
            # convert frame to floats for libsms
            frame = audio[sample_offset:sample_offset + size_new_data]
            frame = np.array(frame, dtype=np.float32)
            analysis_data = pysms.SMS_Data()
            pysms.sms_allocFrameH(sms_header, analysis_data)
            status = pysms.sms_analyze(frame, analysis_data, analysis_params)  

            if status == 1:
                num_partials = analysis_data.nTracks
                sms_freqs = np.zeros(num_partials, dtype=np.float32)
                sms_amps = np.zeros(num_partials, dtype=np.float32)
                sms_phases = np.zeros(num_partials, dtype=np.float32)
                analysis_data.getSinFreq(sms_freqs)
                analysis_data.getSinAmp(sms_amps)
                analysis_data.getSinPhase(sms_phases)
                # make partial objects
                for i in range(num_partials):
                    # for each partial, if the mag is > 0, this partial is alive
                    if sms_amps[i] > 0:
                        # create a peak object
                        p = simpl.Peak()
                        p.amplitude = sms_amps[i]
                        p.frequency = sms_freqs[i]
                        p.phase = sms_phases[i]
                        # add this peak to the appropriate partial
                        if not live_partials[i]:
                            live_partials[i] = simpl.Partial()
                            live_partials[i].starting_frame = current_frame
                            sms_partials.append(live_partials[i])
                        live_partials[i].add_peak(p)
                    # if the mag is 0 and this partial was alive, kill it
                    else:
                        if live_partials[i]:
                            live_partials[i] = None
            elif status == -1:
                do_analysis = False
            pysms.sms_freeFrame(analysis_data)
            current_frame += 1

        pysms.sms_freeAnalysis(analysis_params)
        pysms.sms_closeSF()
        pysms.sms_free()

        pd = simpl.SMSPeakDetection()
        pd.max_peaks = self.max_peaks
        pd.hop_size = self.hop_size 
        peaks = pd.find_peaks(audio)[0:self.num_frames]
        pt = simpl.SMSPartialTracking()
        pt.max_partials = self.max_peaks
        partials = pt.find_partials(peaks)

        import debug
        debug.print_partials(sms_partials)
        print
        debug.print_partials(partials)
        #raise Exception("ok")

        # make sure both have the same number of partials
        self.assertEquals(len(sms_partials), len(partials))

        # make sure each partial is the same
        for i in range(len(sms_partials)):
            self.assertEquals(sms_partials[i].get_length(), partials[i].get_length())
            for peak_number in range(sms_partials[i].get_length()):
                self.assertAlmostEquals(sms_partials[i].peaks[peak_number].amplitude,
                                        partials[i].peaks[peak_number].amplitude,
                                        places = self.FLOAT_PRECISION)
                self.assertAlmostEquals(sms_partials[i].peaks[peak_number].frequency,
                                        partials[i].peaks[peak_number].frequency,
                                        places = self.FLOAT_PRECISION)
                self.assertAlmostEquals(sms_partials[i].peaks[peak_number].phase,
                                        partials[i].peaks[peak_number].phase,
                                        places = self.FLOAT_PRECISION)

   #def test_interpolate_frames(self):
    #    """test_interpolate_frames
    #    Make sure that pysms.sms_interpolateFrames returns the expected values
    #    with interpolation factors of 0 and 1."""
    #    pysms.sms_init()
    #    sms_header = pysms.SMS_Header()
    #    snd_header = pysms.SMS_SndHeader()
    #    # Try to open the input file to fill snd_header
    #    if(pysms.sms_openSF(input_file, snd_header)):
    #        raise NameError("error opening sound file: " + pysms.sms_errorString())
    #    analysis_params = pysms.SMS_AnalParams()
    #    analysis_params.iSamplingRate = 44100
    #    analysis_params.iFrameRate = sampling_rate / hop_size
    #    sms_header.nStochasticCoeff = 128
    #    analysis_params.fDefaultFundamental = 100
    #    analysis_params.fHighestFreq = 20000
    #    analysis_params.iMaxDelayFrames = 3
    #    analysis_params.analDelay = 0
    #    analysis_params.minGoodFrames = 1
    #    analysis_params.iFormat = pysms.SMS_FORMAT_HP
    #    analysis_params.nTracks = max_partials
    #    analysis_params.nGuides = max_partials
    #    analysis_params.iWindowType = pysms.SMS_WIN_HAMMING
    #    pysms.sms_initAnalysis(analysis_params, snd_header)
    #    analysis_params.nFrames = num_samples / hop_size
    #    analysis_params.iSizeSound = num_samples
    #    analysis_params.peakParams.iMaxPeaks = max_peaks
    #    analysis_params.iStochasticType = pysms.SMS_STOC_NONE
    #    pysms.sms_fillHeader(sms_header, analysis_params, "pysms")
    #    interp_frame = pysms.SMS_Data()
    #    pysms.sms_allocFrame(interp_frame, sms_header.nTracks, sms_header.nStochasticCoeff, 1, sms_header.iStochasticType, 0)

    #    sample_offset = 0
    #    size_new_data = 0
    #    current_frame = 0
    #    sms_header.nFrames = num_frames
    #    analysis_frames = []
    #    do_analysis = True

    #    while do_analysis and (current_frame < num_frames):
    #        sample_offset += size_new_data
    #        if((sample_offset + analysis_params.sizeNextRead) < num_samples):
    #            size_new_data = analysis_params.sizeNextRead
    #        else:
    #            size_new_data = num_samples - sample_offset
    #        frame = audio[sample_offset:sample_offset + size_new_data]
    #        analysis_data = pysms.SMS_Data()
    #        pysms.sms_allocFrameH(sms_header, analysis_data)
    #        status = pysms.sms_analyze(frame, analysis_data, analysis_params)  

    #        if status == 1:
    #            analysis_frames.append(analysis_data)
    #            # test interpolateFrames on the last two analysis frames
    #            if current_frame == num_frames - 1:
    #                left_frame = analysis_frames[-2]
    #                right_frame = analysis_frames[-1]
    #                pysms.sms_interpolateFrames(left_frame, right_frame, interp_frame, 0)
    #                # make sure that interp_frame == left_frame
    #                # interpolateFrames doesn't interpolate phases so ignore
    #                left_amps = simpl.zeros(max_partials)
    #                left_freqs = simpl.zeros(max_partials)
    #                left_frame.getSinAmp(left_amps)
    #                left_frame.getSinFreq(left_freqs)
    #                right_amps = simpl.zeros(max_partials)
    #                right_freqs = simpl.zeros(max_partials)
    #                right_frame.getSinAmp(right_amps)
    #                right_frame.getSinFreq(right_freqs)
    #                interp_amps = simpl.zeros(max_partials)
    #                interp_freqs = simpl.zeros(max_partials)
    #                interp_frame.getSinAmp(interp_amps)
    #                interp_frame.getSinFreq(interp_freqs)
    #                for i in range(max_partials):
    #                    self.assertAlmostEquals(left_amps[i], interp_amps[i],
    #                                            places = FLOAT_PRECISION)
    #                    if left_freqs[i] != 0:
    #                        self.assertAlmostEquals(left_freqs[i], interp_freqs[i],
    #                                                places = FLOAT_PRECISION)
    #                    else:
    #                        self.assertAlmostEquals(right_freqs[i], interp_freqs[i],
    #                                                places = FLOAT_PRECISION)
    #                pysms.sms_interpolateFrames(left_frame, right_frame, interp_frame, 1)
    #                interp_amps = simpl.zeros(max_partials)
    #                interp_freqs = simpl.zeros(max_partials)
    #                interp_frame.getSinAmp(interp_amps)
    #                interp_frame.getSinFreq(interp_freqs)
    #                for i in range(max_partials):
    #                    self.assertAlmostEquals(right_amps[i], interp_amps[i],
    #                                            places = FLOAT_PRECISION)
    #                    if right_freqs[i] != 0:
    #                        self.assertAlmostEquals(right_freqs[i], interp_freqs[i],
    #                                                places = FLOAT_PRECISION)
    #                    else:
    #                        self.assertAlmostEquals(left_freqs[i], interp_freqs[i],
    #                                                places = FLOAT_PRECISION)
    #        elif status == -1:
    #            raise Exception("AnalysisStoppedEarly")
    #        current_frame += 1

    #    pysms.sms_freeAnalysis(analysis_params)
    #    pysms.sms_closeSF()

    #def test_harmonic_synthesis(self):
    #    """test_harmonic_synthesis
    #    Compare pysms synthesised harmonic component with SMS synthesised 
    #    harmonic component."""
    #    pysms.sms_init()
    #    sms_header = pysms.SMS_Header()
    #    snd_header = pysms.SMS_SndHeader()
    #    # Try to open the input file to fill snd_header
    #    if(pysms.sms_openSF(input_file, snd_header)):
    #        raise NameError("error opening sound file: " + pysms.sms_errorString())
    #    analysis_params = pysms.SMS_AnalParams()
    #    analysis_params.iSamplingRate = 44100
    #    analysis_params.iFrameRate = sampling_rate / hop_size
    #    sms_header.nStochasticCoeff = 128
    #    analysis_params.fDefaultFundamental = 100
    #    analysis_params.fHighestFreq = 20000
    #    analysis_params.iMaxDelayFrames = 3
    #    analysis_params.analDelay = 0
    #    analysis_params.minGoodFrames = 1
    #    analysis_params.iFormat = pysms.SMS_FORMAT_HP
    #    analysis_params.nTracks = max_partials
    #    analysis_params.nGuides = max_partials
    #    analysis_params.iWindowType = pysms.SMS_WIN_HAMMING
    #    pysms.sms_initAnalysis(analysis_params, snd_header)
    #    analysis_params.nFrames = num_samples / hop_size
    #    analysis_params.iSizeSound = num_samples
    #    analysis_params.peakParams.iMaxPeaks = max_peaks
    #    analysis_params.iStochasticType = pysms.SMS_STOC_NONE
    #    pysms.sms_fillHeader(sms_header, analysis_params, "pysms")

    #    sample_offset = 0
    #    size_new_data = 0
    #    current_frame = 0
    #    sms_header.nFrames = num_frames
    #    analysis_frames = []
    #    do_analysis = True

    #    while do_analysis and (current_frame < num_frames):
    #        sample_offset += size_new_data
    #        if((sample_offset + analysis_params.sizeNextRead) < num_samples):
    #            size_new_data = analysis_params.sizeNextRead
    #        else:
    #            size_new_data = num_samples - sample_offset
    #        frame = audio[sample_offset:sample_offset + size_new_data]
    #        analysis_data = pysms.SMS_Data()
    #        pysms.sms_allocFrameH(sms_header, analysis_data)
    #        status = pysms.sms_analyze(frame, analysis_data, analysis_params)  
    #        analysis_frames.append(analysis_data)
    #        if status == -1:
    #            do_analysis = False
    #        current_frame += 1

    #    pysms.sms_freeAnalysis(analysis_params)
    #    pysms.sms_closeSF()

    #    interp_frame = pysms.SMS_Data() 
    #    synth_params = pysms.SMS_SynthParams() 
    #    synth_params.iSynthesisType = pysms.SMS_STYPE_DET
    #    synth_params.iDetSynthType = pysms.SMS_DET_SIN
    #    synth_params.sizeHop = hop_size 
    #    synth_params.iSamplingRate = 0

    #    pysms.sms_initSynth(sms_header, synth_params)
    #    pysms.sms_allocFrame(interp_frame, sms_header.nTracks, sms_header.nStochasticCoeff, 1, sms_header.iStochasticType, sms_header.nEnvCoeff)

    #    synth_samples = pysms.zeros(synth_params.sizeHop)
    #    num_synth_samples = 0
    #    target_synth_samples = len(analysis_frames) * hop_size
    #    pysms_audio = pysms.array([])
    #    current_frame = 0

    #    while num_synth_samples < target_synth_samples:
    #        pysms.sms_synthesize(analysis_frames[current_frame], synth_samples, synth_params)
    #        pysms_audio = np.hstack((pysms_audio, synth_samples))
    #        num_synth_samples += synth_params.sizeHop
    #        current_frame += 1

    #    pysms.sms_freeSynth(synth_params)
    #    pysms.sms_free()     

    #    pd = simpl.SMSPeakDetection()
    #    pd.max_peaks = max_peaks
    #    pd.hop_size = hop_size 
    #    pt = simpl.SMSPartialTracking()
    #    pt.max_partials = max_partials
    #    peaks = pd.find_peaks(audio)
    #    partials = pt.find_partials(peaks[0:num_frames])
    #    synth = simpl.SMSSynthesis()
    #    synth.hop_size = hop_size
    #    synth.stochastic_type = pysms.SMS_STOC_NONE
    #    synth.synthesis_type = pysms.SMS_STYPE_DET
    #    synth.max_partials = max_partials
    #    simpl_audio = synth.synth(partials)

    #    self.assertEquals(pysms_audio.size, simpl_audio.size)
    #    for i in range(simpl_audio.size):
    #        self.assertAlmostEquals(pysms_audio[i], simpl_audio[i],
    #                                places = FLOAT_PRECISION)

    #def test_residual_synthesis(self):
    #    """test_residual_synthesis
    #    Compare pysms residual signal with SMS residual""" 
    #    pysms.sms_init()
    #    sms_header = pysms.SMS_Header()
    #    snd_header = pysms.SMS_SndHeader()
    #    # Try to open the input file to fill snd_header
    #    if(pysms.sms_openSF(input_file, snd_header)):
    #        raise NameError("error opening sound file: " + pysms.sms_errorString())
    #    analysis_params = pysms.SMS_AnalParams()
    #    analysis_params.iSamplingRate = 44100
    #    analysis_params.iFrameRate = sampling_rate / hop_size
    #    sms_header.nStochasticCoeff = 128
    #    analysis_params.fDefaultFundamental = 100
    #    analysis_params.fHighestFreq = 20000
    #    analysis_params.iMaxDelayFrames = 3
    #    analysis_params.analDelay = 0
    #    analysis_params.minGoodFrames = 1
    #    analysis_params.iFormat = pysms.SMS_FORMAT_HP
    #    analysis_params.nTracks = max_partials
    #    analysis_params.nGuides = max_partials
    #    analysis_params.iWindowType = pysms.SMS_WIN_HAMMING
    #    pysms.sms_initAnalysis(analysis_params, snd_header)
    #    analysis_params.nFrames = num_samples / hop_size
    #    analysis_params.iSizeSound = num_samples
    #    analysis_params.peakParams.iMaxPeaks = max_peaks
    #    analysis_params.iStochasticType = pysms.SMS_STOC_APPROX
    #    pysms.sms_fillHeader(sms_header, analysis_params, "pysms")

    #    sample_offset = 0
    #    size_new_data = 0
    #    current_frame = 0
    #    sms_header.nFrames = num_frames
    #    analysis_frames = []
    #    do_analysis = True

    #    while do_analysis and (current_frame < num_frames-1):
    #        sample_offset += size_new_data
    #        if((sample_offset + analysis_params.sizeNextRead) < num_samples):
    #            size_new_data = analysis_params.sizeNextRead
    #        else:
    #            size_new_data = num_samples - sample_offset
    #        frame = audio[sample_offset:sample_offset + size_new_data]
    #        analysis_data = pysms.SMS_Data()
    #        pysms.sms_allocFrameH(sms_header, analysis_data)
    #        status = pysms.sms_analyze(frame, analysis_data, analysis_params)  
    #        analysis_frames.append(analysis_data)
    #        if status == -1:
    #            do_analysis = False
    #        current_frame += 1

    #    pysms.sms_freeAnalysis(analysis_params)
    #    pysms.sms_closeSF()
    #    pysms.sms_free()

    #    pd = simpl.SMSPeakDetection()
    #    pd.max_peaks = max_peaks
    #    pd.hop_size = hop_size
    #    pt = simpl.SMSPartialTracking()
    #    pt.max_partials = max_partials
    #    peaks = pd.find_peaks(audio)
    #    partials = pt.find_partials(peaks[0:num_frames])
    #    synth = simpl.SMSSynthesis()
    #    synth.hop_size = hop_size
    #    synth.stochastic_type = pysms.SMS_STOC_NONE
    #    synth.synthesis_type = pysms.SMS_STYPE_DET
    #    synth.max_partials = max_partials
    #    simpl_harmonic = synth.synth(partials)
    #    res = simpl.SMSResidual()
    #    res.num_coefficients = 128
    #    res.type = simpl.SMSResidual.TIME_DOMAIN
    #    residual = res.find_residual(simpl_harmonic, audio[0:simpl_harmonic.size])
#        print_partials(partials)
#        print simpl_harmonic.size
#        for i in range(residual.size):
#            print residual[i]
#        for i in range(simpl_harmonic.size):
#            print simpl_harmonic[i]
#        from pylab import plot, show
#        plot(simpl_harmonic)
#        plot(residual)
#        plot(audio[0:simpl_harmonic.size])
#        show()
#        from scipy.io.wavfile import write
#        write("res.wav", 44100, residual)
#        res.synth(simpl_harmonic, audio)

if __name__ == "__main__":
    # run individual tests programatically
    # useful for debugging, particularly with GDB
    import nose
    argv = [__file__, 
            #__file__ + ":TestSimplSMS.test_partial_tracking",
            __file__ + ":TestSimplSMS.test_partial_tracking"]
    nose.run(argv=argv)

