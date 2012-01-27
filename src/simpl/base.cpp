#include <iostream>
#include "base.h"

using namespace std;

namespace Simpl {

// ---------------------------------------------------------------------------
// Peak
// ---------------------------------------------------------------------------
Peak::Peak()
{
    amplitude = 0.0;
    frequency = 0.0;
    phase = 0.0;
    next_peak = NULL;
    previous_peak = NULL;
    partial_id = 0;
    partial_position = 0;
    frame_number = 0;
}

Peak::~Peak()
{
}

// Returns true iff this peak is unmatched in the given direction, and has positive amplitude
bool Peak::is_free(const string direction) 
{
    if(amplitude <= 0.0)
    {
        return false;
    }

    if(direction == "forwards")
    {
        if(next_peak != NULL)
        {
            return false;
        }
    }
    else if(direction == "backwards")
    {
        if(previous_peak != NULL)
        {
            return false;
        }
    }
    else
    {
        Throw(InvalidArgument, "Invalid direction");
    }

    return true;
}

// ---------------------------------------------------------------------------
// Frame
// ---------------------------------------------------------------------------
Frame::Frame()
{
    _size = 512;
    init();
}

Frame::Frame(int frame_size)
{
    _size = frame_size;
    init();
}

Frame::~Frame()
{
    _peaks.clear();
    _partials.clear();
}

void Frame::init()
{
    _max_peaks = 100;
    _max_partials = 100;
    _audio = NULL;
    // _synth = NULL;
    // _residual = NULL;
    // _synth_residual = NULL;
}

// Frame - peaks
// -------------

int Frame::num_peaks()
{
    return _peaks.size();
}

int Frame::max_peaks()
{
    return _max_peaks;
}

void Frame::max_peaks(int new_max_peaks)
{
    _max_peaks = new_max_peaks;

    // potentially losing data here but the user shouldn't really do this
    if((int)_peaks.size() > _max_peaks)
    {
        _peaks.resize(_max_peaks);
    }
}

void Frame::add_peak(Peak peak)
{
    _peaks.push_back(peak);
}

void Frame::add_peaks(Peaks* peaks)
{
    for(Peaks::iterator i = peaks->begin(); i != peaks->end(); i++)
    {
        add_peak(Peak(*i));
    }
}

Peak Frame::peak(int peak_number)
{
    return _peaks[peak_number];
}

void Frame::clear_peaks()
{
    _peaks.clear();
}

Peaks::iterator Frame::peaks_begin()
{
    return _peaks.begin();
}

Peaks::iterator Frame::peaks_end()
{
    return _peaks.end();
}

// Frame - partials
// ----------------

int Frame::num_partials()
{
    return _partials.size();
}

int Frame::max_partials()
{
    return _max_partials;
}

void Frame::max_partials(int new_max_partials)
{
    _max_partials = new_max_partials;

    // potentially losing data here but the user shouldn't really do this
    if((int)_partials.size() > _max_partials)
    {
        _partials.resize(_max_partials);
    }
}

void Frame::add_partial(Partial partial)
{

}

Partials::iterator Frame::partials()
{
    return _partials.begin();
}


// Frame - audio buffers
// ---------------------

int Frame::size()
{
    return _size;
}

void Frame::size(int new_size)
{
    _size = new_size;
}

// void Frame::audio(const number* new_audio)
// {
//     _audio = new_audio;
// }

void Frame::audio(const samples& new_audio, const int offset)
{
    _audio = &new_audio + offset;
}

const samples* Frame::audio()
{
    return _audio;
}

// void Frame::synth(const number* new_synth)
// {
//     _synth = new_synth;
// }

// const number* Frame::synth()
// {
//     return _synth;
// }

// void Frame::residual(const number* new_residual)
// {
//     _residual = new_residual;
// }

// const number* Frame::residual()
// {
//     return _residual;
// }

// void Frame::synth_residual(const number* new_synth_residual)
// {
//     _synth_residual = new_synth_residual;
// }

// const number* Frame::synth_residual()
// {
//     return _synth_residual;
// }

// ---------------------------------------------------------------------------
// PeakDetection
// ---------------------------------------------------------------------------

PeakDetection::PeakDetection()
{
    _sampling_rate = 44100;
    _frame_size = 2048;
    _static_frame_size = true;
    _hop_size = 512;
    _max_peaks = 100;
    _window_type = "hamming";
    _window_size = 2048;
    _min_peak_separation = 1.0; // in Hz
}

PeakDetection::~PeakDetection()
{
    _frames.clear();
}

int PeakDetection::sampling_rate()
{
    return _sampling_rate;
}

void PeakDetection::sampling_rate(int new_sampling_rate)
{
    _sampling_rate = new_sampling_rate;
}

int PeakDetection::frame_size()
{
    return _frame_size;
}
void PeakDetection::frame_size(int new_frame_size)
{
    _frame_size = new_frame_size;
}

bool PeakDetection::static_frame_size()
{
    return _static_frame_size;
}

void PeakDetection::static_frame_size(bool new_static_frame_size)
{
    _static_frame_size = new_static_frame_size;
}

int PeakDetection::next_frame_size()
{
    return _frame_size;
}

int PeakDetection::hop_size()
{
    return _hop_size;
}

void PeakDetection::hop_size(int new_hop_size)
{
    _hop_size = new_hop_size;
}

int PeakDetection::max_peaks()
{
    return _max_peaks;
}

void PeakDetection::max_peaks(int new_max_peaks)
{
    _max_peaks = new_max_peaks;
}

std::string PeakDetection::window_type()
{
    return _window_type;
}

void PeakDetection::window_type(std::string new_window_type)
{
    _window_type = new_window_type;
}

int PeakDetection::window_size()
{
    return _window_size;
}

void PeakDetection::window_size(int new_window_size)
{
    _window_size = new_window_size;
}

number PeakDetection::min_peak_separation()
{
    return _min_peak_separation;
}

void PeakDetection::min_peak_separation(number new_min_peak_separation)
{
    _min_peak_separation = new_min_peak_separation;
}

Frames* PeakDetection::frames()
{
    return &_frames;
}

// Find and return all spectral peaks in a given frame of audio
Peaks* PeakDetection::find_peaks_in_frame(const Frame& frame)
{
    Peaks* peaks = new Peaks();
    return peaks;
}

// Find and return all spectral peaks in a given audio signal.
// If the signal contains more than 1 frame worth of audio, it will be broken
// up into separate frames, each containing a std::vector of peaks.
Frames* PeakDetection::find_peaks(const samples& audio)
{
    _frames.clear();
    unsigned int pos = 0;
    while(pos < audio.size())
    {
        // get the next frame size
        if(!_static_frame_size)
        {
            _frame_size = next_frame_size();
        }
        
        // get the next frame
        Frame f = Frame(_frame_size);
        f.audio(audio, pos);

        // find peaks
        Peaks* peaks = find_peaks_in_frame(f);
        f.add_peaks(peaks);
        delete peaks;

        _frames.push_back(f);
        pos += _hop_size;
    }

    return &_frames;
}

} // end of namespace Simpl
