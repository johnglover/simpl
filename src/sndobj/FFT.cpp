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
// FFT.cpp : implementation of the FFT class
//           short-time fast fourier transform
//           Victor Lazzarini, 2003
/////////////////////////////////////////////////
#include "FFT.h"

FFT::FFT(){
  m_table = 0;

  // hopsize controls decimation
  // we have vecsize/hopsize overlapping frames
  // the vector size is also equal the fftsize
  // so that each call to DoProcess produces a
  // new fft frame at the output
  // since SndObj has already allocated the output
  // we have to reset the vector size

  m_fftsize = DEF_FFTSIZE;
  SetVectorSize(DEF_FFTSIZE);
  m_hopsize = DEF_VECSIZE;

  m_frames = m_fftsize/m_hopsize;

  m_sigframe = new double*[m_frames];
  m_counter = new int[m_frames];
  m_halfsize = m_fftsize/2;
  m_fund = m_sr/m_fftsize;
  int i;
  for(i = 0; i < m_frames; i++){
    m_sigframe[i] = new double[m_fftsize];
    memset(m_sigframe[i], 0, m_fftsize*sizeof(double));
    m_counter[i] = i*m_hopsize;
  }

  m_fftIn = (double*) fftw_malloc(sizeof(double) * m_fftsize);
  m_fftOut = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m_fftsize);
  m_plan = fftw_plan_dft_r2c_1d(m_fftsize, m_fftIn, m_fftOut, FFTW_ESTIMATE);
  memset(m_fftIn, 0, m_fftsize*sizeof(double));


  AddMsg("scale", 21);
  AddMsg("fft size", 22);
  AddMsg("hop size", 23);
  AddMsg("window", 24);
  m_scale = 1.f;
  m_norm = m_fftsize;
  m_cur =0;
}

FFT::FFT(Table* window, SndObj* input, double scale,
     int fftsize, int hopsize, double sr):
  SndObj(input, fftsize, sr){
  m_table = window;

  m_hopsize = hopsize;
  m_fftsize = fftsize;
  m_frames = m_fftsize/m_hopsize;

  m_sigframe = new double*[m_frames];
  m_counter = new int[m_frames];
  m_halfsize = m_fftsize/2;
  m_fund = m_sr/m_fftsize;
  int i;
  for(i = 0; i < m_frames; i++){
    m_sigframe[i] = new double[m_fftsize];
    memset(m_sigframe[i], 0, m_fftsize*sizeof(double));
    m_counter[i] = i*m_hopsize;
  }

  m_fftIn = (double*) fftw_malloc(sizeof(double) * m_fftsize);
  m_fftOut = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m_fftsize);
  m_plan = fftw_plan_dft_r2c_1d(m_fftsize, m_fftIn, m_fftOut, FFTW_ESTIMATE);
  memset(m_fftIn, 0, m_fftsize*sizeof(double));

  AddMsg("scale", 21);
  AddMsg("fft size", 22);
  AddMsg("hop size", 23);
  AddMsg("window", 24);
  m_scale = scale;
  m_norm = m_fftsize/m_scale;
  m_cur =0;
}

FFT::~FFT(){
  fftw_destroy_plan(m_plan);
  fftw_free(m_fftIn);
  fftw_free(m_fftOut);
  delete[] m_counter;
  delete[] m_sigframe;
}

void
FFT::SetFFTSize(int fftsize){
  SetVectorSize(m_fftsize = fftsize);
  ReInit();
}

void
FFT::SetHopSize(int hopsize){
  m_hopsize = hopsize;
  ReInit();
}

void
FFT::ReInit(){
  fftw_destroy_plan(m_plan);
  fftw_free(m_fftIn);
  fftw_free(m_fftOut);

  delete[] m_counter;
  delete[] m_sigframe;
  delete[] m_output;

  if(!(m_output = new double[m_vecsize])){
    m_error = 1;
#ifdef DEBUG
    cout << ErrorMessage();
#endif
    return;
  }

  m_frames = m_fftsize/m_hopsize;
  m_sigframe = new double*[m_frames];
  m_counter = new int[m_frames];
  m_halfsize = m_fftsize/2;
  m_fund = m_sr/m_fftsize;
  int i;
  for(i = 0; i < m_frames; i++){
    m_sigframe[i] = new double[m_fftsize];
    memset(m_sigframe[i], 0, m_fftsize*sizeof(double));
    m_counter[i] = i*m_hopsize;
  }

  m_fftIn = (double*) fftw_malloc(sizeof(double) * m_fftsize);
  m_fftOut = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m_fftsize);
  m_plan = fftw_plan_dft_r2c_1d(m_fftsize, m_fftIn, m_fftOut, FFTW_ESTIMATE);
  memset(m_fftIn, 0, m_fftsize*sizeof(double));

  m_cur =0;
  m_norm = m_fftsize/m_scale;
}

int
FFT::Set(const char* mess, double value){
  switch(FindMsg(mess)){

  case 21:
    SetScale(value);
    return 1;

  case 22:
    SetFFTSize((int) value);
    return 1;

  case 23:
    SetHopSize((int) value);
    return 1;

  default:
    return  SndObj::Set(mess, value);
  }
}

int
FFT::Connect(const char* mess, void *input){
  switch(FindMsg(mess)){

  case 24:
    SetWindow((Table *) input);
    return 1;

  default:
    return SndObj::Connect(mess, input);

  }
}

short
FFT::DoProcess(){
  if(!m_error){
    if(m_input && m_table){
      if(m_enable){
        int i; double sig = 0.f;

        for(m_vecpos = 0; m_vecpos < m_hopsize; m_vecpos++){
          // signal input
          sig = m_input->Output(m_vecpos);
          // distribute to the signal fftframes and apply the window
          // according to a time pointer (kept by counter[n])
          for(i=0;i < m_frames; i++){
            m_sigframe[i][m_counter[i]]= sig*m_table->Lookup(m_counter[i]);
            m_counter[i]++;
          }
        }

        // every hopsize samples
        // set the current sigframe to be transformed
        m_cur--;
        if(m_cur<0) m_cur = m_frames-1;

        // transform it and fill the output buffer
        fft(m_sigframe[m_cur]);

        // zero the current sigframe time pointer
        m_counter[m_cur] = 0;
        return 1;

      }
      else{ // if disabled
        for(m_vecpos=0; m_vecpos < m_hopsize; m_vecpos++)
        m_output[m_vecpos] = 0.f;
        return 1;
      }
    }
    else {
      m_error = 3;
      return 0;
    }
  }
  else
    return 0;
}

void
FFT::fft(double* signal){
  memcpy(m_fftIn, &signal[0], sizeof(double) * m_fftsize);
  fftw_execute(m_plan);

  m_output[0] = m_fftOut[0][0] / m_norm;
  m_output[1] = m_fftOut[0][1] / m_norm;

  int i = 2;
  for(int bin = 1; bin < m_halfsize; bin++){
    m_output[i] = (m_fftOut[bin][0] * 2) / m_norm;
    m_output[i+1] = (m_fftOut[bin][1] * 2) / m_norm;
    i += 2;
  }
}
