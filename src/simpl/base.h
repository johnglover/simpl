#ifndef BASE_H
#define BASE_H
/*
 * Copyright (c) 2009-2011 John Glover, National University of Ireland, Maynooth
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */

#include <vector>
#include <string>

#include "exceptions.h"

using namespace std;

namespace Simpl 
{

typedef double number;
typedef std::vector<number> samples;

// ---------------------------------------------------------------------------
// Peak
//
// A spectral Peak
// ---------------------------------------------------------------------------
class Peak
{
public:
    number amplitude;
    number frequency;
    number phase;
    Peak* next_peak;
    Peak* previous_peak;
    int partial_id;
    int partial_position;
    int frame_number;

    Peak();
    ~Peak();

    bool is_start_of_partial()
    {
        return previous_peak == NULL;
    };
    bool is_free(string direction = string("forwards"));
};

typedef std::vector<Peak> Peaks;

// ---------------------------------------------------------------------------
// Partial
// ---------------------------------------------------------------------------
class Partial {};

typedef std::vector<Partial> Partials;

// ---------------------------------------------------------------------------
// Frame
// 
// Represents a frame of audio information.
// This can be: - raw audio samples 
//              - an unordered list of sinusoidal peaks 
//              - an ordered list of partials
//              - synthesised audio samples
//              - residual samples
//              - synthesised residual samples
// ---------------------------------------------------------------------------
class Frame
{
protected:
    int _size;
    int _max_peaks;
    int _max_partials;
    void init();

public:
    Peaks peaks;
    Partials partials;
    number* audio;
    number* synth;
    number* residual;
    number* synth_residual;

    Frame();
    Frame(int frame_size);
    ~Frame();
    int size();
    void size(int new_size);
    int max_peaks();
    void max_peaks(int new_max_peaks);
    int max_partials();
    void max_partials(int new_max_partials);
};

typedef std::vector<Frame> Frames;

// ---------------------------------------------------------------------------
// PeakDetection
// 
// Detect spectral peaks
// ---------------------------------------------------------------------------

class PeakDetection
{
protected:
    int _sampling_rate;
    int _frame_size;
    bool _static_frame_size;
    int _hop_size;
    int _max_peaks;
    std::string _window_type;
    int _window_size;
    number _min_peak_separation;
    Frames _frames;

public:
    PeakDetection();
    ~PeakDetection();

    int sampling_rate();
    void sampling_rate(int new_sampling_rate);
    int frame_size();
    void frame_size(int new_frame_size);
    bool static_frame_size();
    void static_frame_size(bool new_static_frame_size);
    int next_frame_size();
    int hop_size();
    void hop_size(int new_hop_size);
    int max_peaks();
    void max_peaks(int new_max_peaks);
    std::string window_type();
    void window_type(std::string new_window_type);
    int window_size();
    void window_size(int new_window_size);
    number min_peak_separation();
    void min_peak_separation(number new_min_peak_separation);
    Frames* frames();
};

} // end of namespace Simpl

#endif
