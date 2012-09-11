#include "exceptions.h"
#include <string>

using namespace simpl;

Exception::Exception(const std::string & str) : _msg(str) {
}
