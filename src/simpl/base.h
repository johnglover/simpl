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

#include "exceptions.h"

namespace Simpl 
{

typedef double number;

// A Spectral Peak
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
    bool is_free(const char* direction="forwards");
};

typedef std::vector<Peak> Peaks;

} // end of namespace Simpl

#endif
