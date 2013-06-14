#include <iostream>
#include "portaudio.h"
#include "simpl/simpl.h"

using namespace std;

#define SAMPLE_RATE 44100


class AnalysisData {
public:
    const int max_peaks;
    std::vector<double> audio;
    simpl::Frame frame;
    simpl::LorisPeakDetection pd;
    simpl::SMSPartialTracking pt;
    simpl::SMSSynthesis synth;

    AnalysisData(int frame_size, int hop_size) : max_peaks(50),
                                                 frame(frame_size, true) {
        audio.resize(hop_size);

        frame.synth_size(hop_size);
        frame.max_peaks(max_peaks);
        frame.max_partials(max_peaks);

        pd.frame_size(frame_size);
        pd.hop_size(hop_size);
        pd.max_peaks(max_peaks);

        pt.realtime(true);
        pt.max_partials(max_peaks);

        synth.det_synthesis_type(0);
        synth.hop_size(hop_size);
        synth.max_partials(frame_size);
    }
};


static int patestCallback(const void *input,
                          void *output,
                          unsigned long buffer_size,
                          const PaStreamCallbackTimeInfo* time_info,
                          PaStreamCallbackFlags status_flags,
                          void *frame_data) {
    AnalysisData* data = (AnalysisData*)frame_data;
    float *in = (float*)input;
    float *out = (float*)output;

    std::copy(in, in + buffer_size, data->audio.begin());
    data->frame.audio(&(data->audio[0]), buffer_size);

    data->pd.find_peaks_in_frame(&(data->frame));
    data->pt.update_partials(&(data->frame));
    data->synth.synth_frame(&(data->frame));

    for(unsigned int i = 0; i < buffer_size; i++) {
        out[i] = data->frame.synth()[i];
    }

    data->frame.clear_peaks();
    data->frame.clear_partials();
    data->frame.clear_synth();
    return 0;
}


int main() {
    PaError err;
    PaStream *stream;
    int n_input_chans = 1;
    int n_output_chans = 1;
    int buffer_size = 512;
    int frame_size = 2048;
    static AnalysisData data(frame_size, buffer_size);

    err = Pa_Initialize();
    if(err != paNoError) {
        cout << "Error initialising PortAudio" << endl;
        return 1;
    }

    err = Pa_OpenDefaultStream(&stream,
                               n_input_chans,
                               n_output_chans,
                               paFloat32,
                               SAMPLE_RATE,
                               buffer_size,
                               patestCallback,
                               &data);
    if(err != paNoError) {
        cout << "Error opening default audio stream" << endl;
        return 1;
    }

    err = Pa_StartStream(stream);
    if(err != paNoError) {
        cout << "Error starting audio stream" << endl;
        return 1;
    }

    cout << endl;
    cout << "Analysing audio from default input and "
         << "synthesising to default output." << endl;
    cout << "Press Enter to stop" << endl;
    cin.ignore();
    
    err = Pa_StopStream(stream);
    if(err != paNoError) {
        cout << "Error stopping audio stream" << endl;
        return 1;
    }

    err = Pa_CloseStream(stream);
    if(err != paNoError) {
        cout << "Error closing audio stream" << endl;
        return 1;
    }

    err = Pa_Terminate();
    if(err != paNoError) {
        cout << "Error terminating PortAudio" << endl;
        return 1;
    }

    return 0;
}
