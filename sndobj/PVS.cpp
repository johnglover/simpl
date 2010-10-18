 
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

/////////////////////////////////////////////////
// PVS.cpp : Phase Vocoder Synthesis Class
//
//           Victor Lazzarini, 2003
//
/////////////////////////////////////////////////
#include "PVS.h"


PVS::PVS(){
  m_rotcount = m_vecsize;
  m_phases = new float[m_halfsize];
  memset(m_phases, 0, sizeof(float)*m_halfsize);
  m_factor = (m_hopsize*TWOPI)/m_sr;
  m_first = true;
}

PVS::PVS(Table* window, SndObj* input, int fftsize,
	 int hopsize, float sr)
  :IFFT(window, input,fftsize,hopsize,sr)
{
  m_rotcount = m_vecsize;
  if(m_halfsize){
    m_phases = new float[m_halfsize];
    memset(m_phases, 0, sizeof(float)*m_halfsize);
  }
  m_factor = (m_hopsize*TWOPI)/m_sr;
  m_first = true;
}

PVS::~PVS(){
  if(m_halfsize) 
    delete[] m_phases;
}


int
PVS::Set(char* mess, float value){
	
  switch(FindMsg(mess)){
		
  case 22:
    SetFFTSize((int) value);
    return 1;
			
  case 23:
    SetHopSize((int) value);
    return 1;
			
  default:
    return	IFFT::Set(mess, value);
			
  }
}

void
PVS::SetFFTSize(int fftsize){
  m_rotcount = m_vecsize;
  IFFT::SetFFTSize(fftsize);
}

void
PVS::SetHopSize(int hopsize){
  m_rotcount = m_vecsize;
  m_factor = (m_hopsize*TWOPI)/m_sr;
  IFFT::SetFFTSize(hopsize);
}

void
PVS::pvsynthesis(float* signal){
  double pha;
  int i2;
	
  m_ffttmp[0] = m_input->Output(0); 
  m_ffttmp[m_halfsize] = m_input->Output(1);
	
  for(int i=0;i<m_fftsize; i+=2){ 
    i2 = i/2;
    m_phases[i2] += m_input->Output(i+1) - m_fund*i2; 
    pha = m_phases[i2]*m_factor;
    m_ffttmp[i2] = m_input->Output(i)*cos(pha);
    m_ffttmp[m_fftsize-(i2)] = m_input->Output(i)*sin(pha);
  }
	
  rfftw_one(m_plan, m_ffttmp, signal);
}


short
PVS::DoProcess(){
  if(!m_error){
    if(m_input){
      if(m_enable){
	int i; float out = 0.;
	// phase vocoder synthesis
				
	if(m_first) {
	  for(m_vecpos = 0; m_vecpos < m_vecsize; m_vecpos++)
	    m_output[m_vecpos] = 0.f;
	} else {
	  pvsynthesis(m_sigframe[m_cur]);
	}
				
	// set the current signal frame to the next
	// one in the circular list
	m_counter[m_cur] = 0;	    
	m_cur++; if(m_cur==m_frames) m_cur = 0;
				
				
	for(m_vecpos = 0; m_vecpos < m_vecsize; m_vecpos++){ 
	  // overlap-add the time-domain signal frames
	  // also make sure the frames are unrotated
	  for(i=0; i < m_frames; i++){
	    out += m_sigframe[i][m_rotcount]*m_table->Lookup(m_counter[i]);
	    m_counter[i]++;
	  }
	  m_rotcount++;
	  // output it.
	  m_output[m_vecpos] = (float) out;
	  out = 0.;	   
	}
	m_rotcount %= m_fftsize;
	m_first = false;
	return 1;
      } else { // if disabled
	for(m_vecpos = 0; m_vecpos < m_vecsize; m_vecpos++)
	  m_output[m_vecpos] = 0.f;
	m_first = true;
	return 1;
      }
    } else {
      m_error = 3;
      return 0;
    }
  }
  else 
    return 0;
}








