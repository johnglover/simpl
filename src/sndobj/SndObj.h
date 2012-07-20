
////////////////////////////////////////////////////////////////////////
// This file is part of the SndObj library
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//
// Copyright (c)Victor Lazzarini, 1997-2004
// See License.txt for a disclaimer of all warranties
// and licensing information

//************************************************************//
//  SndObj.h: Interface of the SndObj base class              //
//                                                            //
//                                                            //
//                                                            //
//************************************************************//

#ifndef _SNDOBJ_H
#define _SNDOBJ_H
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <iostream>
#include <string>
#ifndef WIN
#include <unistd.h>
#endif

using namespace std;

class  SndIO;

// PI is defined in a few places (rfftw, sms, etc) so check that it's
// not already defined
#ifndef PI
const double PI = 4.*atan(1.);
#endif

const int DEF_FFTSIZE = 1024;
const int DEF_VECSIZE = 256;
const double DEF_SR = 44100.f;

struct msg_link {
  string msg;
  int  ID;
  msg_link *previous;
};

class SndObj {

 protected:

  double* m_output; // output samples
  SndObj* m_input; // input object
  double  m_sr;    // sampling rate
  int    m_vecsize; //vector size
  int    m_vecpos; // vector pos counter
  int    m_vecsize_max; // for limiting operation
  int    m_altvecpos; // secondary counter
  int    m_error;     // error code
  short  m_enable;  // enable object

  msg_link *m_msgtable;

  inline int FindMsg(const char* mess);
  void AddMsg(const char* mess, int ID);

#if defined (WIN) && !defined(GCC)
  int Ftoi(double x){
    union {
      double   f;
      int     i;
    } u;
    unsigned int  tmp;
    unsigned char   tmp2;
    u.f = x;
    tmp2 = (unsigned char) 158 - (unsigned char) (((int) u.i & 0x7F800000) >> 23);
    if (tmp2 & (unsigned char) 0xE0)
      return (unsigned int) 0;
    tmp = (unsigned int) u.i | (unsigned int) 0xFF800000UL;
    tmp = (tmp << 8) >> tmp2;
    return (u.i < (int) 0 ? -((int) tmp) : (int) tmp);
  }

  int Ftoi(double fval){
    int temp;
    short oldcw;
    short tempcw;
    _asm {
      fnstcw  oldcw      /*save current control reg*/
	wait
	mov     ax,oldcw
	or      ah,0Ch     /*set truncation mode     */
	mov     tempcw,ax
	fldcw   tempcw
	fld     fval       /*do the conversion...    */
	fistp   temp
	fldcw   oldcw      /*restore register        */
	mov     eax,temp   /* = "return temp;"       */
        }
    return temp;
  }
#else
  int Ftoi(double fval) { return (int) fval; }
  int Ftoi(float fval) { return (int) fval; }
#endif

 public:

  bool IsProcessing() {
    if(m_vecpos && m_vecpos != m_vecsize) return true;
    else return false;
  }

  int GetError() { return m_error; }

#ifndef SWIGJAVA
  SndObj operator=(SndObj obj){
    if(&obj == this) return *this;
    for(int n = 0; n < m_vecsize; n++) m_output[n] = obj.Output(n);
    return *this;
  }

  SndObj& operator+=(SndObj& obj){
    for(int n = 0; n < m_vecsize; n++) m_output[n] = m_output[n]+obj.Output(n);
    return *this;
  }

  SndObj& operator-=(SndObj& obj){
    for(int n = 0; n < m_vecsize; n++) m_output[n] = m_output[n]-obj.Output(n);
    return *this;
  }

  SndObj& operator*=(SndObj& obj){
    for(int n = 0; n < m_vecsize; n++) m_output[n] = m_output[n]*obj.Output(n);
    return *this;
  }

  SndObj& operator+=(double val){
    for(int n = 0; n < m_vecsize; n++) m_output[n] = m_output[n]+val;
    return *this;
  }

  SndObj& operator-=(double val){
    for(int n = 0; n < m_vecsize; n++) m_output[n] = m_output[n]-val;
    return *this;
  }

  SndObj& operator*=(double val){
    for(int n = 0; n < m_vecsize; n++) m_output[n] = m_output[n]*val;
    return *this;
  }

  SndObj operator+(SndObj& obj){
    SndObj temp(0, m_vecsize, m_sr);
    for(int n = 0; n < m_vecsize; n++) temp.m_output[n] = m_output[n]+obj.Output(n);
    return temp;
  }

  SndObj operator-(SndObj& obj){
    SndObj temp(0, m_vecsize, m_sr);
    for(int n = 0; n < m_vecsize; n++) temp.m_output[n] = m_output[n]-obj.Output(n);
    return temp;
  }

  SndObj operator*(SndObj& obj){
    SndObj temp(0, m_vecsize, m_sr);
    for(int n = 0; n < m_vecsize; n++) temp.m_output[n] = m_output[n]*obj.Output(n);
    return temp;
  }

  SndObj operator+(double val){
    SndObj temp(0, m_vecsize, m_sr);
    for(int n = 0; n < m_vecsize; n++) temp.m_output[n] = m_output[n]+val;
    return temp;
  }

  SndObj operator-(double val){
    SndObj temp(0, m_vecsize, m_sr);
    for(int n = 0; n < m_vecsize; n++) temp.m_output[n] = m_output[n]-val;
    return temp;
  }

  SndObj operator*(double val){
    SndObj temp(0, m_vecsize, m_sr);
    for(int n = 0; n < m_vecsize; n++) temp.m_output[n] = m_output[n]*val;
    return temp;
  }

  void operator<<(double val){
    if(m_vecpos >= m_vecsize) m_vecpos=0;
    m_output[m_vecpos++] = val;
  }

  void operator<<(double* vector){
    for(m_vecpos=0;m_vecpos<m_vecsize;m_vecpos++)
      m_output[m_vecpos] = vector[m_vecpos];
  }

  void operator>>(SndIO& out);
  void operator<<(SndIO& in);

#endif

  int PushIn(double *in_vector, int size){
    for(int i = 0; i<size; i++){
      if(m_vecpos >= m_vecsize) m_vecpos = 0;
      m_output[m_vecpos++] = in_vector[i];
    }
    return m_vecpos;
  }

  int PopOut(double *out_vector, int size){
    for(int i = 0; i<size; i++){
      if(m_altvecpos >= m_vecsize) m_altvecpos = 0;
      out_vector[i] = m_output[m_altvecpos++];
    }
    return m_altvecpos;
  }


  int AddOut(double *vector, int size){
    for(int i = 0; i<size; i++){
      if(m_altvecpos >= m_vecsize) m_altvecpos = 0;
      vector[i] += m_output[m_altvecpos++];
    }
    return m_altvecpos;
  }


  void GetMsgList(string* list);
  void Enable(){ m_enable = 1; }
  void Disable(){ m_enable = 0; }
  virtual double Output(int pos){ return m_output[pos%m_vecsize]; }

  int GetVectorSize() { return m_vecsize; }
  void SetVectorSize(int vecsize);
  void LimitVectorSize(int limit) {
        if(limit <= m_vecsize_max)
                 m_vecsize = limit;
  }
  void RestoreVectorSize(){ m_vecsize = m_vecsize_max; }
  double GetSr(){ return m_sr;}
  virtual void SetSr(double sr){ m_sr = sr;}
  virtual int Set(const char* mess, double value);
  virtual int Connect(const char* mess, void* input);


  void SetInput(SndObj* input){
    m_input = input;
  }

  SndObj* GetInput(){ return m_input; }

  SndObj(SndObj* input, int vecsize = DEF_VECSIZE, double sr = DEF_SR);
  SndObj();
#if !defined (SWIGPYTHON) && !defined(SWIGCFFI)
  SndObj(SndObj& obj);
#endif

  virtual ~SndObj();
  virtual const char* ErrorMessage();
  virtual short DoProcess();
};

int
SndObj::FindMsg(const char* mess){
  msg_link* iter = m_msgtable;
  while(iter->previous && iter->msg.compare(mess))
    iter = iter->previous;
  if(!iter->msg.compare(mess)) return iter->ID;
  else return 0;
}


#endif
