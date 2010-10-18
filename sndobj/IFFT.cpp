 
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
// IFFT.cpp: implementation of the IFFT class
//           short-time inverse fast fourier transform
//           Victor Lazzarini, 2003
/////////////////////////////////////////////////
#include "IFFT.h"


IFFT::IFFT(){

  m_table = 0;

  // vectorsize equals hopsize
  // so that each call of DoProcess adds a
  // new window to the overlap-add process

  m_hopsize = DEF_VECSIZE;
  m_fftsize = DEF_FFTSIZE;
  m_frames = m_fftsize/m_hopsize;

  m_sigframe = new float*[m_frames];
  m_ffttmp = new float[m_fftsize];
  m_counter = new int[m_frames];
  m_halfsize = m_fftsize/2;
  m_fund = m_sr/m_fftsize;
  int i;
  memset(m_ffttmp, 0, m_fftsize*sizeof(float));
  for(i = 0; i < m_frames; i++){
    m_sigframe[i] = new float[m_fftsize];
    memset(m_sigframe[i], 0, m_fftsize*sizeof(float));
    m_counter[i] = i*m_hopsize;
  }

  m_plan = rfftw_create_plan(m_fftsize, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE);

  AddMsg("fft size", 21);
  AddMsg("hop size", 22);
  AddMsg("window", 23);

  m_cur = 0;
}

IFFT::IFFT(Table* window, SndObj* input, int fftsize, int hopsize,
	   float sr):
  SndObj(input,hopsize, sr)
{
	
  m_table = window;
	
  m_hopsize = hopsize;
  m_fftsize = fftsize;	
	
  if(m_fftsize){
    m_frames = m_fftsize/m_hopsize;

    m_sigframe = new float*[m_frames];
    m_ffttmp = new float[m_fftsize];
    m_counter = new int[m_frames];
    m_halfsize = m_fftsize/2;
    m_fund = m_sr/m_fftsize;
    memset(m_ffttmp, 0, m_fftsize*sizeof(float));
    int i;
    for(i = 0; i < m_frames; i++){
      m_sigframe[i] = new float[m_fftsize];
      memset(m_sigframe[i], 0, m_fftsize*sizeof(float));
      m_counter[i] = i*m_hopsize;
    }
		
    m_plan = rfftw_create_plan(m_fftsize, FFTW_COMPLEX_TO_REAL, FFTW_ESTIMATE);
  }
	
  AddMsg("fft size", 21);
  AddMsg("hop size", 22);
  AddMsg("window", 23);
	
  m_cur = 0;
}

IFFT::~IFFT(){
	
  if(m_fftsize){
#ifndef WIN
    rfftw_destroy_plan(m_plan);
#endif
    delete[] m_counter;
    delete[] m_ffttmp;
    delete[] m_sigframe;
  }
}

void
IFFT::SetFFTSize(int fftsize){
  m_fftsize = fftsize;
  ReInit();
}

void
IFFT::SetHopSize(int hopsize){
  SetVectorSize(m_hopsize = hopsize);
  ReInit();
}

void 
IFFT::ReInit(){	

  rfftw_destroy_plan(m_plan);
  delete[] m_counter;
  delete[] m_sigframe;
  delete[] m_ffttmp;
  delete[] m_output;

  if(!(m_output = new float[m_vecsize])){
    m_error = 1;
#ifdef DEBUG
    cout << ErrorMessage();
#endif
    return;
  }


  m_frames = m_fftsize/m_hopsize;
  m_sigframe = new float*[m_frames];
  m_ffttmp = new float[m_fftsize];
  m_counter = new int[m_frames];
  m_halfsize = m_fftsize/2;
  m_fund = m_sr/m_fftsize;
  int i;
  for(i = 0; i < m_frames; i++){
    m_sigframe[i] = new float[m_fftsize];
    memset(m_sigframe[i], 0, m_fftsize*sizeof(float));
    m_counter[i] = i*m_hopsize;
  }

  m_plan = rfftw_create_plan(m_vecsize, FFTW_REAL_TO_COMPLEX, FFTW_ESTIMATE);
  m_cur =0;

}


int
IFFT::Set(char* mess, float value){

  switch(FindMsg(mess)){

  case 21:
    SetFFTSize((int) value);
    return 1;

  case 22:
    SetHopSize((int) value);
    return 1;
	
  default:
    return	SndObj::Set(mess, value);

  }
}

int
IFFT::Connect(char* mess, void *input){

  switch(FindMsg(mess)){

  case 23:
    SetWindow((Table *) input);
    return 1;

  default:
    return SndObj::Connect(mess, input);

  }

}

short
IFFT::DoProcess(){


  if(!m_error){
    if(m_input && m_table){
      if(m_enable){
	int i; float out = 0.;  	  	  
	// Put the input fftframe into
	// the current (free) signal frame
	// and transform it
	ifft(m_sigframe[m_cur]);
	// set the current signal frame to the next
	// one in the circular list
	m_counter[m_cur] = 0;
	m_cur--; if(m_cur<0) m_cur = m_frames-1;	 
		    
	for(m_vecpos = 0; m_vecpos < m_vecsize; m_vecpos++){ 
	  // overlap-add the time-domain signal frames
	  for(i=0; i < m_frames; i++){
	    out += m_sigframe[i][m_counter[i]]*m_table->Lookup(m_counter[i]);
	    m_counter[i]++;
	  }
	  // output it.
	  m_output[m_vecpos] = (float) out;
	  out = 0.;	   
	}

	return 1;
      } else { // if disabled
	for(m_vecpos = 0; m_vecpos < m_vecsize; m_vecpos++)
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




void 
IFFT::ifft(float* signal) {

  // get input FFT frame and
  // prepare data for fftw

  m_ffttmp[0] = m_input->Output(0); 
  m_ffttmp[m_halfsize] = m_input->Output(1);
  for(int i=2, i2=1; i<m_fftsize; i+=2){
    i2 = i/2;
    m_ffttmp[i2] = m_input->Output(i);
    m_ffttmp[m_fftsize-(i2)] = m_input->Output(i+1);
  }	

  // Inverse FFT function
  rfftw_one(m_plan, m_ffttmp, signal);

}


