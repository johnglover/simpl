%include exception.i 
%include std_string.i
%include std_vector.i
%include std_list.i
%include "numpy.i"

%template(DoubleVector) std::vector<Simpl::number>;

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
