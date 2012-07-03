import os
import json
import numpy as np
import scipy.io.wavfile as wav
import pysms
import simpl

float_precision = 5
frame_size = 512
hop_size = 512
max_peaks = 10
max_partials = 10
num_frames = 30
num_samples = num_frames * hop_size
audio_path = os.path.join(
    os.path.dirname(__file__), 'audio/flute.wav'
)

audio, sampling_rate = simpl.read_wav(audio_path)


def _pysms_analysis_params(sampling_rate):
    analysis_params = pysms.SMS_AnalParams()
    pysms.sms_initAnalParams(analysis_params)
    analysis_params.iSamplingRate = sampling_rate
    analysis_params.iFrameRate = sampling_rate / hop_size
    analysis_params.iWindowType = pysms.SMS_WIN_HAMMING
    analysis_params.fHighestFreq = 20000
    analysis_params.iFormat = pysms.SMS_FORMAT_HP
    analysis_params.nTracks = max_peaks
    analysis_params.peakParams.iMaxPeaks = max_peaks
    analysis_params.nGuides = max_peaks
    analysis_params.iMaxDelayFrames = 4
    analysis_params.analDelay = 0
    analysis_params.minGoodFrames = 1
    analysis_params.iCleanTracks = 0
    analysis_params.iStochasticType = pysms.SMS_STOC_NONE
    analysis_params.preEmphasis = 0
    return analysis_params


def _pysms_synthesis_params(sampling_rate):
    synth_params = pysms.SMS_SynthParams()
    pysms.sms_initSynthParams(synth_params)
    synth_params.iSamplingRate = sampling_rate
    synth_params.iSynthesisType = pysms.SMS_STYPE_DET
    synth_params.iStochasticType = pysms.SMS_STOC_NONE
    synth_params.sizeHop = hop_size
    synth_params.nTracks = max_partials
    synth_params.deEmphasis = 0
    return synth_params


def _size_next_read():
    pysms.sms_init()
    snd_header = pysms.SMS_SndHeader()

    # Try to open the input file to fill snd_header
    if(pysms.sms_openSF(audio_path, snd_header)):
        raise NameError(
            "error opening sound file: " + pysms.sms_errorString()
        )

    analysis_params = _pysms_analysis_params(sampling_rate)
    analysis_params.iMaxDelayFrames = num_frames + 1
    if pysms.sms_initAnalysis(analysis_params, snd_header) != 0:
        raise Exception("Error allocating memory for analysis_params")
    analysis_params.nFrames = num_frames
    sms_header = pysms.SMS_Header()
    pysms.sms_fillHeader(sms_header, analysis_params, "pysms")

    sample_offset = 0
    pysms_size_new_data = 0
    current_frame = 0
    next_read_sizes = []

    while current_frame < num_frames:
        next_read_sizes.append(analysis_params.sizeNextRead)
        sample_offset += pysms_size_new_data
        pysms_size_new_data = analysis_params.sizeNextRead

        # convert frame to floats for libsms
        frame = audio[sample_offset:sample_offset + pysms_size_new_data]
        frame = np.array(frame, dtype=np.float32)
        if len(frame) < pysms_size_new_data:
            frame = np.hstack((
                frame, np.zeros(pysms_size_new_data - len(frame),
                                dtype=np.float32)
            ))

        analysis_data = pysms.SMS_Data()
        pysms.sms_allocFrameH(sms_header, analysis_data)
        status = pysms.sms_analyze(frame, analysis_data, analysis_params)
        # as the no. of frames of delay is > num_frames, sms_analyze should
        # never get around to performing partial tracking, and so the
        # return value should be 0
        assert status == 0
        pysms.sms_freeFrame(analysis_data)
        current_frame += 1

    pysms.sms_freeAnalysis(analysis_params)
    pysms.sms_closeSF()
    pysms.sms_free()

    return next_read_sizes


def _partial_tracking():
    pysms.sms_init()
    snd_header = pysms.SMS_SndHeader()

    if(pysms.sms_openSF(audio_path, snd_header)):
        raise NameError(pysms.sms_errorString())

    analysis_params = _pysms_analysis_params(sampling_rate)
    if pysms.sms_initAnalysis(analysis_params, snd_header) != 0:
        raise Exception("Error allocating memory for analysis_params")
    analysis_params.iSizeSound = num_samples
    analysis_params.nFrames = num_frames
    sms_header = pysms.SMS_Header()
    pysms.sms_fillHeader(sms_header, analysis_params, "pysms")

    sample_offset = 0
    size_new_data = 0
    current_frame = 0
    sms_frames = []
    do_analysis = True

    while do_analysis and (current_frame < num_frames):
        sample_offset += size_new_data
        size_new_data = analysis_params.sizeNextRead

        frame_audio = audio[sample_offset:sample_offset + size_new_data]
        frame_audio = np.array(frame_audio, dtype=np.float32)
        if len(frame_audio) < size_new_data:
            frame_audio = np.hstack((
                frame_audio, np.zeros(size_new_data - len(frame_audio),
                                      dtype=np.float32)
            ))

        analysis_data = pysms.SMS_Data()
        pysms.sms_allocFrameH(sms_header, analysis_data)
        num_partials = analysis_data.nTracks
        status = pysms.sms_analyze(frame_audio, analysis_data,
                                   analysis_params)

        sms_freqs = np.zeros(num_partials, dtype=np.float32)
        sms_amps = np.zeros(num_partials, dtype=np.float32)
        sms_phases = np.zeros(num_partials, dtype=np.float32)

        frame = {'status': status}
        frame['partials'] = []

        if status == 1:
            analysis_data.getSinFreq(sms_freqs)
            analysis_data.getSinAmp(sms_amps)
            analysis_data.getSinPhase(sms_phases)
            current_frame += 1

        if status == -1:
            do_analysis = False

        for i in range(num_partials):
            frame['partials'].append({
                'n': i,
                'amplitude': float(sms_amps[i]),
                'frequency': float(sms_freqs[i]),
                'phase': float(sms_phases[i])
            })

        sms_frames.append(frame)
        pysms.sms_freeFrame(analysis_data)

    pysms.sms_freeAnalysis(analysis_params)
    pysms.sms_closeSF()
    pysms.sms_free()

    return sms_frames


def _harmonic_synthesis(det_synth_type):
    pysms.sms_init()
    snd_header = pysms.SMS_SndHeader()

    if(pysms.sms_openSF(audio_path, snd_header)):
        raise NameError(pysms.sms_errorString())

    analysis_params = _pysms_analysis_params(sampling_rate)
    if pysms.sms_initAnalysis(analysis_params, snd_header) != 0:
        raise Exception("Error allocating memory for analysis_params")
    analysis_params.iSizeSound = num_samples
    analysis_params.nFrames = num_frames
    sms_header = pysms.SMS_Header()
    pysms.sms_fillHeader(sms_header, analysis_params, "pysms")

    sample_offset = 0
    size_new_data = 0
    current_frame = 0
    analysis_frames = []
    do_analysis = True

    while do_analysis and (current_frame < num_frames):
        sample_offset += size_new_data
        size_new_data = analysis_params.sizeNextRead

        frame_audio = audio[sample_offset:sample_offset + size_new_data]
        frame_audio = np.array(frame_audio, dtype=np.float32)
        if len(frame_audio) < size_new_data:
            frame_audio = np.hstack((
                frame_audio, np.zeros(size_new_data - len(frame_audio),
                                      dtype=np.float32)
            ))

        analysis_data = pysms.SMS_Data()
        pysms.sms_allocFrameH(sms_header, analysis_data)
        status = pysms.sms_analyze(frame_audio, analysis_data,
                                   analysis_params)

        analysis_frames.append(analysis_data)
        current_frame += 1

        if status == -1:
            do_analysis = False

    synth_params = _pysms_synthesis_params(sampling_rate)
    if det_synth_type == 'ifft':
        synth_params.iDetSynthType = pysms.SMS_DET_IFFT
    elif det_synth_type == 'sin':
        synth_params.iDetSynthType = pysms.SMS_DET_SIN
    else:
        raise Exception("Invalid deterministic synthesis type")

    pysms.sms_initSynth(sms_header, synth_params)

    synth_frame = np.zeros(synth_params.sizeHop, dtype=np.float32)
    synth_audio = np.array([], dtype=np.float32)

    for i in range(len(analysis_frames)):
        pysms.sms_synthesize(analysis_frames[i], synth_frame, synth_params)
        synth_audio = np.hstack((synth_audio, synth_frame))

    synth_audio = np.asarray(synth_audio * 32768, np.int16)

    for frame in analysis_frames:
        pysms.sms_freeFrame(frame)
    pysms.sms_freeAnalysis(analysis_params)
    pysms.sms_closeSF()
    pysms.sms_freeSynth(synth_params)
    pysms.sms_free()

    return synth_audio


if __name__ == '__main__':
    size_next_read = _size_next_read()
    partial_tracking = _partial_tracking()
    harmonic_synthesis_ifft = _harmonic_synthesis('ifft')
    harmonic_synthesis_sin = _harmonic_synthesis('sin')

    test_data = {'size_next_read': size_next_read,
                 'peak_detection': partial_tracking,
                 'partial_tracking': partial_tracking}

    test_data = json.dumps(test_data)
    with open('libsms_test_data.json', 'w') as f:
        f.write(test_data)

    wav.write('libsms_harmonic_synthesis_ifft.wav', sampling_rate,
              harmonic_synthesis_ifft)
    wav.write('libsms_harmonic_synthesis_sin.wav', sampling_rate,
              harmonic_synthesis_sin)
