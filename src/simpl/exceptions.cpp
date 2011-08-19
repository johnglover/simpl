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

#include "exceptions.h"
#include <string>

namespace Simpl {

// ---------------------------------------------------------------------------
//	Exception constructor
// ---------------------------------------------------------------------------
//! Construct a new instance with the specified description and, optionally
//! a string identifying the location at which the exception as thrown. The
//! Throw(Exception_Class, description_string) macro generates a location
//! string automatically using __FILE__ and __LINE__.
//!
//! \param  str is a string describing the exceptional condition
//! \param  where is an option string describing the location in
//!         the source code from which the exception was thrown
//!         (generated automatically byt he Throw macro).
//
Exception::Exception(const std::string & str, const std::string & where) :
	_sbuf(str)
{
	_sbuf.append(where);
	_sbuf.append(" ");
}
	
// ---------------------------------------------------------------------------
//	append 
// ---------------------------------------------------------------------------
//! Append the specified string to this Exception's description,
//! and return a reference to this Exception.
//! 
//! \param  str is text to append to the exception description
//! \return a reference to this Exception.
//
Exception & 
Exception::append(const std::string & str)
{
	_sbuf.append(str);
	return *this;
}

} // end of namespace Simpl
