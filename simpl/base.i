%{
	#include "../src/simpl/base.h"
	#include "../src/simpl/exceptions.h"
    #include <vector>
	#define SWIG_FILE_WITH_INIT
%}

%include exception.i 
%include typemaps.i 
%include std_string.i
%include std_vector.i
%include std_list.i
%include "numpy.i"

%init 
%{
    import_array();
%}

%template(DoubleVector) std::vector<Simpl::number>;

%include "../src/simpl/base.h"
%include "../src/simpl/exceptions.h"

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

%extend Simpl::Frame
{
    Simpl::samples get_audio()
    {
        if($self->audio())
        {
            /* const Simpl::samples* sp = $self->audio(); */
            /* std::cout << sp->at(0) << std::endl; */

            /* Simpl::samples s = Simpl::samples(*($self->audio())); */
            /* return s; */
            return Simpl::samples(512, 0.5);
        }
        else
        {
            return Simpl::samples($self->size(), 0.00);
        }
    }

    void set_audio(const samples& new_audio)
    {
        $self->audio(new_audio);
    }

    %pythoncode 
    %{
        __swig_getmethods__["audio"] = get_audio
        __swig_setmethods__["audio"] = set_audio
        if _newclass: audio = property(get_audio, set_audio)
    %}
};
