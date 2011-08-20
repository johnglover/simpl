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

#include "base.h"

using namespace std;

namespace Simpl {

// ---------------------------------------------------------------------------
//	Peak
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
bool Peak::is_free(string direction)
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
//	Frame
// ---------------------------------------------------------------------------
Frame::Frame()
{
    size = 512;
    init();
}

Frame::Frame(int frame_size)
{
    size = frame_size;
    init();
}

Frame::~Frame()
{
}

void Frame::init()
{
    audio.resize(size);
    synth.resize(size);
    residual.resize(size);
    synth_residual.resize(size);
    max_peaks = 100;
    peaks.resize(max_peaks);
    max_partials = 100;
    partials.resize(max_partials);
}

int Frame::get_size()
{
    return size;
}

int Frame::get_max_peaks()
{
    return max_peaks;
}

int Frame::get_max_partials()
{
    return max_partials;
}

void Frame::set_size(int new_size)
{
    size = new_size;
    audio.resize(size);
    synth.resize(size);
    residual.resize(size);
    synth_residual.resize(size);
}

void Frame::set_max_peaks(int new_max_peaks)
{
    max_peaks = new_max_peaks;
    peaks.resize(max_peaks);
}

void Frame::set_max_partials(int new_max_partials)
{
    max_partials = new_max_partials;
    partials.resize(max_partials);
}

} // end of namespace Simpl
