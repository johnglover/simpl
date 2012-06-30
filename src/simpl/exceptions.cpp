#include "exceptions.h"
#include <string>

namespace simpl {

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
