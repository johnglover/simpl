 
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
// PVA.cpp: Phase Vocoder Analysis class
//
//           Victor Lazzarini, 2003
/////////////////////////////////////////////////
#include "PVA.h"

PVA::PVA(){
  m_rotcount = 0;
  m_phases = new float[m_halfsize];
  memset(m_phases, 0, sizeof(float)*m_halfsize);
  m_factor = m_sr/(m_hopsize*TWOPI);
}


PVA::PVA(Table* window, SndObj* input, float scale,
	 int fftsize, int hopsize, float sr)
  :FFT(window, input, scale, fftsize, hopsize, sr)
{
  m_rotcount = 0;
  m_phases = new float[m_halfsize];
  memset(m_phases, 0, sizeof(float)*m_halfsize);
  m_factor = m_sr/(m_hopsize*TWOPI);
}


PVA::~PVA(){
  delete[] m_phases;

}


int
PVA::Set(char* mess, float value){

  switch(FindMsg(mess)){

  case 22:
    SetFFTSize((int) value);
    return 1;

  case 23:
    SetHopSize((int) value);
    return 1;
	
  default:
    return	FFT::Set(mess, value);

  }
}
void
PVA::SetFFTSize(int fftsize){
  m_rotcount = 0;
  FFT::SetFFTSize(fftsize);
}

void
PVA::SetHopSize(int hopsize){
  m_rotcount = 0;
  m_factor = m_sr/(hopsize*TWOPI);
  FFT::SetFFTSize(hopsize);
}

void
PVA::pvanalysis(float* signal){

  double re, im, pha, diff; 
  int i2;
  
  rfftw_one(m_plan, signal, m_ffttmp);

  m_output[0] = m_ffttmp[0]/m_norm;
  m_output[1] = m_ffttmp[m_halfsize]/m_norm;
  
  for(int i=2; i<m_fftsize; i+=2){
    i2 = i/2;
    re = m_ffttmp[i2]/m_norm;
    im = m_ffttmp[m_fftsize-(i2)]/m_norm;

    if((m_output[i] = sqrt((re*re)+(im*im)))==0.f){
      diff = 0.f; 
    }
    else {
      pha = atan2(im,re);
      diff = pha - m_phases[i2];
      m_phases[i2] = (float) pha;

      while(diff > PI) diff -= TWOPI;
      while(diff < -PI) diff += TWOPI;
    }
    m_output[i+1] = (float) diff*m_factor + i2*m_fund;

  }

}


short
PVA::DoProcess(){

  if(!m_error){
    if(m_input){
      if(m_enable){
	int i; float sig = 0.f;
	for(m_vecpos = 0; m_vecpos < m_hopsize; m_vecpos++) {
	  // signal input
	  sig = m_input->Output(m_vecpos);		
	  // distribute to the signal fftframes and apply the window
	  // according to a time pointer (kept by counter[n])
	  // input is also rotated according to the input time.
	  for(i=0;i < m_frames; i++){
	    m_sigframe[i][m_rotcount]= (float) sig*m_table->Lookup(m_counter[i]);
	    m_counter[i]++;		   
	  }  
	  m_rotcount++;
	} 
	m_rotcount %= m_fftsize;
	// every vecsize samples
	// set the current fftframe to be transformed
	m_cur--; if(m_cur<0) m_cur = m_frames-1;  
 
	// phase vocoder analysis
	pvanalysis(m_sigframe[m_cur]);

	// zero the current fftframe time pointer
	m_counter[m_cur] = 0;
	return 1;

      } else { // if disabled, reset the fftframes
	for(m_vecpos =0; m_vecpos < m_hopsize; m_vecpos++)
	  m_output[m_vecpos] = 0.f;
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

