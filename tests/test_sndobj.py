import numpy as np
import sndobj
import simpl

float_precision = 2


class TestSimplSndObj:
    pass
    # def test_ifgram(self):
    #     "Compare sndobj.IFGram with simplsndobj.IFGram"
    #     frame_size = 2048
    #     hop_size = 512
    #     num_frames = 4
    #     num_samples = frame_size + ((num_frames - 1) * hop_size)

    #     audio_in_data = read(self.input_file)
    #     audio_in = simpl.asarray(audio_in_data[1]) / 32768.0
    #     audio_in = audio_in[0:num_samples]
    #     frame = simpl.zeros(frame_size)

    #     sndobj_input = sndobj.SndObj()
    #     simpl_input = pysndobj.SndObj()
    #     sndobj_input.SetVectorSize(frame_size)
    #     simpl_input.SetVectorSize(frame_size)
    #     sndobj_window = sndobj.HammingTable(frame_size, 0.5)
    #     simpl_window = pysndobj.HammingTable(frame_size, 0.5)
    #     sndobj_ifgram = sndobj.IFGram(sndobj_window, sndobj_input, 1, frame_size, hop_size)
    #     simpl_ifgram = pysndobj.IFGram(simpl_window, simpl_input, 1, frame_size, hop_size)

    #     i = 0
    #     while (i + frame_size) <= num_samples:
    #         frame = audio_in[i:i+frame_size]
    #         sndobj_input.PushIn(frame)
    #         simpl_input.PushIn(frame)
    #         sndobj_ifgram.DoProcess()
    #         simpl_ifgram.DoProcess()
    #         for j in range(frame_size):
    #             self.assertAlmostEquals(sndobj_ifgram.Output(j), simpl_ifgram.Output(j),
    #                                     places = FLOAT_PRECISION)
    #         i += hop_size

    # def test_sinanal(self):
    #     "Compare sndobj.SinAnal with simplsndobj.SinAnal"
    #     frame_size = 2048
    #     hop_size = 512
    #     max_tracks = 20
    #     num_frames = 4
    #     num_samples = frame_size + ((num_frames - 1) * hop_size)

    #     audio_in_data = read(self.input_file)
    #     audio_in = simpl.asarray(audio_in_data[1]) / 32768.0
    #     audio_in = audio_in[0:num_samples]
    #     frame = simpl.zeros(frame_size)

    #     sndobj_input = sndobj.SndObj()
    #     simpl_input = pysndobj.SndObj()
    #     sndobj_input.SetVectorSize(frame_size)
    #     simpl_input.SetVectorSize(frame_size)
    #     sndobj_window = sndobj.HammingTable(frame_size, 0.5)
    #     simpl_window = pysndobj.HammingTable(frame_size, 0.5)
    #     sndobj_ifgram = sndobj.IFGram(sndobj_window, sndobj_input, 1, frame_size, hop_size)
    #     simpl_ifgram = pysndobj.IFGram(simpl_window, simpl_input, 1, frame_size, hop_size)
    #     sndobj_sinmod = sndobj.SinAnal(sndobj_ifgram, 0.003, max_tracks, 1, 3)
    #     simpl_sinmod = pysndobj.SinAnal(simpl_ifgram, 0.003, max_tracks, 1, 3)

    #     i = 0
    #     while (i + frame_size) <= num_samples:
    #         frame = audio_in[i:i+frame_size]
    #         sndobj_input.PushIn(frame)
    #         simpl_input.PushIn(frame)
    #         sndobj_ifgram.DoProcess()
    #         simpl_ifgram.DoProcess()
    #         sndobj_sinmod.DoProcess()
    #         simpl_sinmod.DoProcess()
    #         for j in range(max_tracks * 3):
    #             self.assertAlmostEquals(sndobj_sinmod.Output(j), simpl_sinmod.Output(j),
    #                                     places = FLOAT_PRECISION)
    #         i += hop_size

    # def test_adsyn_doprocess(self):
    #     "Compare sndobj.AdSyn with simplsndobj.AdSyn"
    #     frame_size = 2048
    #     hop_size = 512
    #     max_tracks = 20
    #     num_frames = 4
    #     num_samples = frame_size + ((num_frames - 1) * hop_size)

    #     audio_in_data = read(self.input_file)
    #     audio_in = simpl.asarray(audio_in_data[1]) / 32768.0
    #     audio_in = audio_in[0:num_samples]
    #     frame = simpl.zeros(frame_size)
    #     sndobj_frame_out = simpl.zeros(hop_size)
    #     simpl_frame_out = simpl.zeros(hop_size)
    #     sndobj_audio_out = simpl.array([])
    #     simpl_audio_out = simpl.array([])

    #     sndobj_input = sndobj.SndObj()
    #     simpl_input = pysndobj.SndObj()
    #     sndobj_input.SetVectorSize(frame_size)
    #     simpl_input.SetVectorSize(frame_size)
    #     sndobj_window = sndobj.HammingTable(frame_size, 0.5)
    #     simpl_window = pysndobj.HammingTable(frame_size, 0.5)
    #     sndobj_ifgram = sndobj.IFGram(sndobj_window, sndobj_input, 1, frame_size, hop_size)
    #     simpl_ifgram = pysndobj.IFGram(simpl_window, simpl_input, 1, frame_size, hop_size)
    #     sndobj_sinmod = sndobj.SinAnal(sndobj_ifgram, 0.003, max_tracks, 1, 3)
    #     simpl_sinmod = pysndobj.SinAnal(simpl_ifgram, 0.003, max_tracks, 1, 3)
    #     sndobj_table = sndobj.HarmTable(10000, 1, 1, 0.25)
    #     simpl_table = pysndobj.HarmTable(10000, 1, 1, 0.25)
    #     sndobj_synth = sndobj.AdSyn(sndobj_sinmod, max_tracks, sndobj_table, 1, 1, hop_size)
    #     simpl_synth = pysndobj.AdSyn(simpl_sinmod, max_tracks, simpl_table, 1, 1, hop_size)

    #     i = 0
    #     while (i + frame_size) <= num_samples:
    #         frame = audio_in[i:i+frame_size]
    #         sndobj_input.PushIn(frame)
    #         simpl_input.PushIn(frame)
    #         sndobj_ifgram.DoProcess()
    #         simpl_ifgram.DoProcess()
    #         sndobj_sinmod.DoProcess()
    #         simpl_sinmod.DoProcess()
    #         sndobj_synth.DoProcess()
    #         simpl_synth.DoProcess()
    #         sndobj_synth.PopOut(sndobj_frame_out)
    #         simpl_synth.PopOut(simpl_frame_out)
    #         sndobj_audio_out = np.hstack((sndobj_audio_out, sndobj_frame_out))
    #         simpl_audio_out = np.hstack((simpl_audio_out, simpl_frame_out))
    #         i += hop_size

    #     self.assertEqual(sndobj_audio_out.size, simpl_audio_out.size)
    #     for i in range(sndobj_audio_out.size):
    #         self.assertAlmostEquals(sndobj_audio_out[i], simpl_audio_out[i],
    #                                 places = FLOAT_PRECISION)

    # def test_sinsyn_doprocess(self):
    #     "Compare sndobj.SinSyn with pysndobj.SinSyn"
    #     frame_size = 2048
    #     hop_size = 512
    #     max_tracks = 20
    #     num_frames = 4
    #     num_samples = frame_size + ((num_frames - 1) * hop_size)

    #     audio_in_data = read(self.input_file)
    #     audio_in = simpl.asarray(audio_in_data[1]) / 32768.0
    #     audio_in = audio_in[0:num_samples]
    #     frame = simpl.zeros(frame_size)
    #     sndobj_frame_out = simpl.zeros(hop_size)
    #     simpl_frame_out = simpl.zeros(hop_size)
    #     sndobj_audio_out = simpl.array([])
    #     simpl_audio_out = simpl.array([])

    #     sndobj_input = sndobj.SndObj()
    #     simpl_input = pysndobj.SndObj()
    #     sndobj_input.SetVectorSize(frame_size)
    #     simpl_input.SetVectorSize(frame_size)
    #     sndobj_window = sndobj.HammingTable(frame_size, 0.5)
    #     simpl_window = pysndobj.HammingTable(frame_size, 0.5)
    #     sndobj_ifgram = sndobj.IFGram(sndobj_window, sndobj_input, 1, frame_size, hop_size)
    #     simpl_ifgram = pysndobj.IFGram(simpl_window, simpl_input, 1, frame_size, hop_size)
    #     sndobj_sinmod = sndobj.SinAnal(sndobj_ifgram, 0.003, max_tracks, 1, 3)
    #     simpl_sinmod = pysndobj.SinAnal(simpl_ifgram, 0.003, max_tracks, 1, 3)
    #     sndobj_table = sndobj.HarmTable(10000, 1, 1, 0.25)
    #     simpl_table = pysndobj.HarmTable(10000, 1, 1, 0.25)
    #     sndobj_synth = sndobj.SinSyn(sndobj_sinmod, max_tracks, sndobj_table, 1, hop_size)
    #     simpl_synth = pysndobj.SinSyn(simpl_sinmod, max_tracks, simpl_table, 1, hop_size)

    #     i = 0
    #     while (i + frame_size) <= num_samples:
    #         frame = audio_in[i:i+frame_size]
    #         sndobj_input.PushIn(frame)
    #         simpl_input.PushIn(frame)
    #         sndobj_ifgram.DoProcess()
    #         simpl_ifgram.DoProcess()
    #         sndobj_sinmod.DoProcess()
    #         simpl_sinmod.DoProcess()
    #         sndobj_synth.DoProcess()
    #         simpl_synth.DoProcess()
    #         sndobj_synth.PopOut(sndobj_frame_out)
    #         simpl_synth.PopOut(simpl_frame_out)
    #         sndobj_audio_out = np.hstack((sndobj_audio_out, sndobj_frame_out))
    #         simpl_audio_out = np.hstack((simpl_audio_out, simpl_frame_out))
    #         i += hop_size

    #     self.assertEqual(sndobj_audio_out.size, simpl_audio_out.size)
    #     for i in range(sndobj_audio_out.size):
    #         self.assertAlmostEquals(sndobj_audio_out[i], simpl_audio_out[i],
    #                                 places = FLOAT_PRECISION)

    # def test_partial_tracking(self):
    #     "Compare pysndobj Partials with SndObj tracks"
    #     frame_size = 2048
    #     hop_size = 512
    #     num_frames = 4
    #     max_tracks = 20
    #     max_peaks = 20
    #     num_samples = frame_size + ((num_frames - 1) * hop_size)

    #     audio_in_data = read(self.input_file)
    #     audio_in = simpl.asarray(audio_in_data[1]) / 32768.0
    #     audio_in = audio_in[0:num_samples]
    #     frame = simpl.zeros(frame_size)

    #     sndobj_input = sndobj.SndObj()
    #     sndobj_input.SetVectorSize(frame_size)
    #     sndobj_window = sndobj.HammingTable(frame_size, 0.5)
    #     sndobj_ifgram = sndobj.IFGram(sndobj_window, sndobj_input, 1, frame_size, hop_size)
    #     sndobj_sinmod = pysndobj.SinAnal(sndobj_ifgram, 0.003, max_tracks, 1, 3)

    #     pd = simpl.SndObjPeakDetection()
    #     pd.max_peaks = max_peaks
    #     pt = simpl.SndObjPartialTracking()
    #     pt.max_partials = max_tracks

    #     i = 0
    #     while (i + frame_size) <= num_samples:
    #         frame = audio_in[i:i+frame_size]
    #         sndobj_input.PushIn(frame)
    #         sndobj_ifgram.DoProcess()
    #         sndobj_sinmod.DoProcess()

    #         frame = audio_in[i:i+frame_size]
    #         peaks = pd.find_peaks(frame)[0]
    #         partials = pt.update_partials(peaks, i/hop_size)

    #         num_sndobj_partials = sndobj_sinmod.GetTracks()
    #         num_simpl_partials = len(partials)
    #         self.assertEquals(num_sndobj_partials, num_simpl_partials)

    #         for j in range(num_simpl_partials):
    #             self.assertAlmostEquals(partials[j].amplitude, sndobj_sinmod.Output(j*3),
    #                                     places=FLOAT_PRECISION)
    #             self.assertAlmostEquals(partials[j].frequency, sndobj_sinmod.Output((j*3)+1),
    #                                     places=FLOAT_PRECISION)
    #             self.assertAlmostEquals(partials[j].phase, sndobj_sinmod.Output((j*3)+2),
    #                                     places=FLOAT_PRECISION)
    #         i += hop_size

    # def test_synthesis(self):
    #     "Compare pysndobj synthesised audio with SndObj synthesised audio"
    #     frame_size = 2048
    #     hop_size = 512
    #     num_frames = 4
    #     max_tracks = 20
    #     max_peaks = 20
    #     num_samples = frame_size + ((num_frames - 1) * hop_size)

    #     audio_in_data = read(self.input_file)
    #     audio_in = simpl.asarray(audio_in_data[1]) / 32768.0
    #     audio_in = audio_in[0:num_samples]
    #     frame = simpl.zeros(frame_size)
    #     sndobj_frame_out = simpl.zeros(hop_size)
    #     sndobj_audio_out = simpl.array([])

    #     sndobj_input = sndobj.SndObj()
    #     sndobj_input.SetVectorSize(frame_size)
    #     sndobj_window = sndobj.HammingTable(frame_size, 0.5)
    #     sndobj_ifgram = sndobj.IFGram(sndobj_window, sndobj_input, 1, frame_size, hop_size)
    #     sndobj_sinmod = pysndobj.SinAnal(sndobj_ifgram, 0.003, max_tracks, 1, 3)
    #     sndobj_table = sndobj.HarmTable(10000, 1, 1, 0.25)
    #     sndobj_synth = sndobj.AdSyn(sndobj_sinmod, max_tracks, sndobj_table, 1, 1, hop_size)

    #     i = 0
    #     while (i + frame_size) <= num_samples:
    #         frame = audio_in[i:i+frame_size]
    #         sndobj_input.PushIn(frame)
    #         sndobj_ifgram.DoProcess()
    #         sndobj_sinmod.DoProcess()
    #         sndobj_synth.DoProcess()
    #         sndobj_synth.PopOut(sndobj_frame_out)
    #         sndobj_audio_out = np.hstack((sndobj_audio_out, sndobj_frame_out))
    #         i += hop_size

    #     pd = simpl.SndObjPeakDetection()
    #     pd.max_peaks = max_peaks
    #     pt = simpl.SndObjPartialTracking()
    #     pt.max_partials = max_tracks
    #     peaks = pd.find_peaks(audio_in)
    #     partials = pt.find_partials(peaks)
    #     synth = simpl.SndObjSynthesis()
    #     simpl_audio_out = synth.synth(partials)
    #     self.assertEquals(sndobj_audio_out.size, simpl_audio_out.size)
    #     for i in range(simpl_audio_out.size):
    #         self.assertAlmostEquals(sndobj_audio_out[i], simpl_audio_out[i],
    #                                 places = FLOAT_PRECISION)
