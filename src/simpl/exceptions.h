#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H
//
// Mostly taken from the LorisException.h file in Loris (http://www.cerlsoundgroup.org/loris)

#include <stdexcept>
#include <string>

namespace simpl {

// ---------------------------------------------------------------------------
//  class Exception
//
//! Exception is a generic exception class for reporting exceptional 
//! circumstances in Simpl. Exception is derived from std:exception, 
//! and is the base for a hierarchy of derived exception classes
//! in Simpl.
//!
//
class Exception : public std::exception
{
public:
    //! Construct a new instance with the specified description and, optionally
    //! a string identifying the location at which the exception as thrown. The
    //! Throw(Exception_Class, description_string) macro generates a location
    //! string automatically using __FILE__ and __LINE__.
    //!
    //! \param  str is a string describing the exceptional condition
    //! \param  where is an option string describing the location in
    //!         the source code from which the exception was thrown
    //!         (generated automatically by the Throw macro).
    Exception(const std::string & str, const std::string & where = "");
     
    //! Destroy this Exception.
    virtual ~Exception(void) throw() {}

    //! Return a description of this Exception in the form of a
    //! C-style string (char pointer). Overrides std::exception::what.
    //!
    //! \return a C-style string describing the exceptional condition.
    const char * what(void) const throw() 
    {
        return _sbuf.c_str();
    }
     
    //! Append the specified string to this Exception's description,
    //! and return a reference to this Exception.
    //! 
    //! \param  str is text to append to the exception description
    //! \return a reference to this Exception.
    Exception & append(const std::string & str);
     
    //! Return a read-only refernce to this Exception's 
    //! description string.
    //!
    //! \return a string describing the exceptional condition
    const std::string & str(void) const 
    { 
        return _sbuf;
    }

protected:
    //! string for storing the exception description
    std::string _sbuf;
    
};

// ----------------------------------------------------------------------------
//  class AssertionFailure
//
//! Class of exceptions thrown when an assertion (usually representing an
//! invariant condition, and usually detected by the Assert macro) is
//! violated.
// 
class AssertionFailure : public Exception
{
public: 

    //! Construct a new instance with the specified description and, optionally
    //! a string identifying the location at which the exception as thrown. The
    //! Throw(Exception_Class, description_string) macro generates a location
    //! string automatically using __FILE__ and __LINE__.
    //!
    //! \param  str is a string describing the exceptional condition
    //! \param  where is an option string describing the location in
    //!         the source code from which the exception was thrown
    //!         (generated automatically by the Throw macro).
    AssertionFailure(const std::string & str, const std::string & where = "") : 
        Exception(std::string("Assertion failed -- ").append(str), where) 
    {
    }
    
};

// ---------------------------------------------------------------------------
//  class IndexOutOfBounds
//
//! Class of exceptions thrown when a subscriptable object is accessed
//! with an index that is out of range.
//
class IndexOutOfBounds : public Exception
{
public: 

    //! Construct a new instance with the specified description and, optionally
    //! a string identifying the location at which the exception as thrown. The
    //! Throw(Exception_Class, description_string) macro generates a location
    //! string automatically using __FILE__ and __LINE__.
    //!
    //! \param  str is a string describing the exceptional condition
    //! \param  where is an option string describing the location in
    //!         the source code from which the exception was thrown
    //!         (generated automatically by the Throw macro).
    IndexOutOfBounds(const std::string & str, const std::string & where = "") : 
        Exception(std::string("Index out of bounds -- ").append(str), where) {}
        
};


// ---------------------------------------------------------------------------
//  class InvalidObject
//
//! Class of exceptions thrown when an object is found to be badly configured
//! or otherwise invalid.
//
class InvalidObject : public Exception
{
public: 

    //! Construct a new instance with the specified description and, optionally
    //! a string identifying the location at which the exception as thrown. The
    //! Throw(Exception_Class, description_string) macro generates a location
    //! string automatically using __FILE__ and __LINE__.
    //!
    //! \param  str is a string describing the exceptional condition
    //! \param  where is an option string describing the location in
    //!         the source code from which the exception was thrown
    //!         (generated automatically by the Throw macro).
    InvalidObject(const std::string & str, const std::string & where = "") : 
        Exception(std::string("Invalid configuration or object -- ").append(str), where) 
    {
    }
    
};

// ---------------------------------------------------------------------------
//  class InvalidIterator
//
//! Class of exceptions thrown when an Iterator is found to be badly configured
//! or otherwise invalid.
//
class InvalidIterator : public InvalidObject
{
public: 

    //! Construct a new instance with the specified description and, optionally
    //! a string identifying the location at which the exception as thrown. The
    //! Throw(Exception_Class, description_string) macro generates a location
    //! string automatically using __FILE__ and __LINE__.
    //!
    //! \param  str is a string describing the exceptional condition
    //! \param  where is an option string describing the location in
    //!         the source code from which the exception was thrown
    //!         (generated automatically by the Throw macro).
    InvalidIterator(const std::string & str, const std::string & where = "") : 
        InvalidObject(std::string("Invalid Iterator -- ").append(str), where) 
    {
    }
    
};

// ---------------------------------------------------------------------------
//  class InvalidArgument
//
//! Class of exceptions thrown when a function argument is found to be invalid.
//
class InvalidArgument : public Exception
{
public: 

    //! Construct a new instance with the specified description and, optionally
    //! a string identifying the location at which the exception as thrown. The
    //! Throw(Exception_Class, description_string) macro generates a location
    //! string automatically using __FILE__ and __LINE__.
    //!
    //! \param  str is a string describing the exceptional condition
    //! \param  where is an option string describing the location in
    //!         the source code from which the exception was thrown
    //!         (generated automatically by the Throw macro).
    InvalidArgument(const std::string & str, const std::string & where = "") : 
        Exception(std::string("Invalid Argument -- ").append(str), where) 
    {
    }
    
};

// ---------------------------------------------------------------------------
//  class RuntimeError
//
//! Class of exceptions thrown when an unanticipated runtime error is 
//! encountered.
//
class RuntimeError : public Exception
{
public: 

    //! Construct a new instance with the specified description and, optionally
    //! a string identifying the location at which the exception as thrown. The
    //! Throw(Exception_Class, description_string) macro generates a location
    //! string automatically using __FILE__ and __LINE__.
    //!
    //! \param  str is a string describing the exceptional condition
    //! \param  where is an option string describing the location in
    //!         the source code from which the exception was thrown
    //!         (generated automatically by the Throw macro).
    RuntimeError(const std::string & str, const std::string & where = "") : 
        Exception(std::string("Runtime Error -- ").append(str), where) 
    {
    }
    
};

// ---------------------------------------------------------------------------
//  class FileIOException
//
//! Class of exceptions thrown when file input or output fails.
//
class FileIOException : public RuntimeError
{
public: 

    //! Construct a new instance with the specified description and, optionally
    //! a string identifying the location at which the exception as thrown. The
    //! Throw(Exception_Class, description_string) macro generates a location
    //! string automatically using __FILE__ and __LINE__.
    //!
    //! \param  str is a string describing the exceptional condition
    //! \param  where is an option string describing the location in
    //!         the source code from which the exception was thrown
    //!         (generated automatically by the Throw macro).
    FileIOException(const std::string & str, const std::string & where = "") : 
        RuntimeError(std::string("File i/o error -- ").append(str), where) 
   {
   }
   
};

// ---------------------------------------------------------------------------
//	macros for throwing exceptions
//
//	The compelling reason for using macros instead of inlines for all these
//	things is that the __FILE__ and __LINE__ macros will be useful.
//
#define __STR(x) __VAL(x)
#define __VAL(x) #x
#define	Throw(exType, report)												\
	throw exType(report, " (" __FILE__ " line: " __STR(__LINE__)") ")

#define Assert(test)														\
	do {																	\
		if (!(test)) Throw(Simpl::AssertionFailure, #test);				    \
	} while (false)

} // end of namespace Simpl

#endif
