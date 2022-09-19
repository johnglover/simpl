#include <string>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <sstream>
#include <algorithm>

template <class T>
inline std::string toString(const T& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

template<class T>
    T fromString(const std::string& s)
{
     std::istringstream stream (s);
     T t;
     stream >> t;
     return t;
}


inline 
void toLower(char *string)
{
    unsigned int i = 0;

    while(string[i] != 0)
    {
        string[i] = (char)tolower(string[i]);
        i++;
    }  
}

inline 
void toUpper(char *string)
{
    unsigned int i = 0;

    while(string[i] != 0)
    {
        string[i] = (char)toupper(string[i]);
        i++;
    }  
}

inline 
string URLEncode(const string &str) 
{
    int len = str.length();
	char* buff = new char[len + 1];
	strcpy(buff,str.c_str());
	string ret = "";

    for(int i=0; i<len; i++) 
    {
		if(IsCharAlphaNum(buff[i])) 
        {
			ret = ret + buff[i];
		}
        else if(buff[i] == ' ') 
        {
			ret = ret + "+";
		}
        else 
        {
			char tmp[6];
			sprintf(tmp ,"%%%x", buff[i]);
			ret = ret + tmp;
		}
	}

	delete[] buff;

	return ret;
}

inline 
bool IsCharAlphaNum(char c) 
{
	char ch = (char)tolower(c);

    if(    ch == 'a' 
        || ch == 'b' 
        || ch == 'c' 
        || ch == 'd' 
        || ch == 'e' 
	    || ch == 'f' 
        || ch == 'g' 
        || ch == 'h' 
        || ch == 'i' 
        || ch == 'j' 
	    || ch == 'k' 
        || ch == 'l' 
        || ch == 'm' 
        || ch == 'n' 
        || ch == 'o' 
	    || ch == 'p' 
        || ch == 'q' 
        || ch == 'r' 
        || ch == 's' 
        || ch == 't' 
	    || ch == 'u' 
        || ch == 'v' 
        || ch == 'w' 
        || ch == 'x' 
        || ch == 'y' 
	    || ch == 'z' 
        || ch == '0' 
        || ch == '1' 
        || ch == '2' 
        || ch == '3' 
	    || ch == '4' 
        || ch == '5' 
        || ch == '6' 
        || ch == '7' 
        || ch == '8' 
	    || ch == '9') 
    {

            return true;
	}

    return false;
}

inline 
string URLDecode(const string &str) 
{
	int len = str.length();
	char* buff = new char[len + 1];
	strcpy(buff,str.c_str());
	string ret = "";

    for(int i=0;i<len;i++) 
    {
		if(buff[i] == '+') 
        {
			ret = ret + " ";
		}
        else if(buff[i] == '%') 
        {
			char tmp[4];
			char hex[4];			
			hex[0] = buff[++i];
			hex[1] = buff[++i];
			hex[2] = '\0';		
			//int hex_i = atoi(hex);
			sprintf(tmp,"%c", ConvertToDec(hex));
			ret = ret + tmp;
		}
        else 
        {
			ret = ret + buff[i];
		}
	}

	delete[] buff;

	return ret;
}

inline 
int ConvertToDec(const char* hex) 
{
	char buff[12];
	sprintf(buff,"%s",hex);
	int ret = 0;
	int len = strlen(buff);

	for(int i=0; i<len; i++) 
    {
		char tmp[4];
        tmp[0] = buff[i];
		tmp[1] = '\0';
		
        GetAsDec(tmp);

		int tmp_i = atoi(tmp);
		
        int rs = 1;
		
        for(int j=i; j<(len-1); j++) 
        {
			rs *= 16;
		}

		ret += (rs * tmp_i);
	}
	return ret;
}

inline 
void GetAsDec(char* hex)
{
	char tmp = (char)tolower(hex[0]);

	if(tmp == 'a') 
    {
		strcpy(hex,"10");
	}
    else if(tmp == 'b') 
    {
		strcpy(hex,"11");
	}
    else if(tmp == 'c') 
    {
		strcpy(hex,"12");
	}
    else if(tmp == 'd') 
    {
		strcpy(hex,"13");
	}
    else if(tmp == 'e') 
    {
		strcpy(hex,"14");
	}
    else if(tmp == 'f') 
    {
		strcpy(hex,"15");
	}
    else if(tmp == 'g') 
    {
		strcpy(hex,"16");
	}
}
