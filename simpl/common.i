%include exception.i 
%include "numpy.i"

%exception 
{
    try
    {   
        $action
    }
    catch(Simpl::Exception & ex) 
    {
        std::string s("Simpl exception: ");
        s.append(ex.what());
        SWIG_exception(SWIG_UnknownError, (char *) s.c_str());
    }
    catch(std::exception & ex) 
    {
        std::string s("std C++ exception: ");
        s.append(ex.what());
        SWIG_exception(SWIG_UnknownError, (char *) s.c_str());
    }
}
