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

} // end of namespace Simpl
