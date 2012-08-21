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
// IFGram.cpp: Instant Freq Analysis class
//
//           Victor Lazzarini, 2003
/////////////////////////////////////////////////

#include "IFGram.h"

IFGram::IFGram(){
  m_diffwin = new double[m_fftsize];
  m_factor = m_sr/TWOPI;
  m_pdiff = new double[m_halfsize];

  m_diffsig = (double*) fftw_malloc(sizeof(double) * m_fftsize);
  m_fftdiff = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m_fftsize);
  m_diffplan = fftw_plan_dft_r2c_1d(m_fftsize, m_diffsig, m_fftdiff, FFTW_ESTIMATE);

  memset(m_diffwin, 0, sizeof(double) * m_fftsize);
  memset(m_pdiff, 0, sizeof(double) * m_halfsize);
  memset(m_diffsig, 0, sizeof(double) * m_fftsize);
}

IFGram::IFGram(Table* window, SndObj* input, double scale,
               int fftsize, int hopsize, double sr)
  :PVA(window, input, scale, fftsize, hopsize, sr){
  m_diffwin = new double[m_fftsize];
  m_pdiff = new double[m_halfsize];
  for(int i=0; i<m_fftsize; i++){
    m_diffwin[i] = m_table->Lookup(i) - m_table->Lookup(i+1);
  }
  m_factor = m_sr/TWOPI;

  m_diffsig = (double*) fftw_malloc(sizeof(double) * m_fftsize);
  m_fftdiff = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m_fftsize);
  m_diffplan = fftw_plan_dft_r2c_1d(m_fftsize, m_diffsig, m_fftdiff, FFTW_ESTIMATE);

  memset(m_pdiff, 0, sizeof(double) * m_halfsize);
  memset(m_diffsig, 0, sizeof(double) * m_fftsize);
}

IFGram::~IFGram(){
  delete[] m_diffwin;

  fftw_destroy_plan(m_diffplan);
  fftw_free(m_diffsig);
  fftw_free(m_fftdiff);
}

int
IFGram::Set(const char* mess, double value){
  switch(FindMsg(mess)){

  case 22:
    SetFFTSize((int) value);
    return 1;

  default:
    return  PVA::Set(mess, value);

  }
}

int
IFGram::Connect(const char* mess, void* input){
  int i;

  switch(FindMsg(mess)){
  case 24:
    SetWindow((Table *) input);
    for(i=0; i<m_fftsize; i++)
      m_diffwin[i] = m_table->Lookup(i) - m_table->Lookup(i+1);
    return 1;

  default:
    return  PVA::Connect(mess,input);
  }
}

void
IFGram::SetFFTSize(int fftsize){
  FFT::SetFFTSize(fftsize);

  delete[] m_diffwin;
  delete[] m_phases;

  fftw_destroy_plan(m_diffplan);
  fftw_free(m_diffsig);
  fftw_free(m_fftdiff);

  m_factor = m_sr*TWOPI/m_fftsize;

  m_diffwin = new double[m_fftsize];
  m_phases = new double[m_halfsize];

  m_diffsig = (double*) fftw_malloc(sizeof(double) * m_fftsize);
  m_fftdiff = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * m_fftsize);

  for(int i=0; i<m_fftsize; i++){
    m_diffwin[i] = m_table->Lookup(i) - m_table->Lookup(i+1);
  }

  memset(m_pdiff, 0, sizeof(double) * m_halfsize);
  memset(m_diffsig, 0, sizeof(double) * m_fftsize);
}

void
IFGram::IFAnalysis(double* signal){
  double powerspec, da,db, a, b, ph,d;

  for(int i = 0; i < m_fftsize; i++){
    m_diffsig[i] = signal[i]*m_diffwin[i];
    signal[i] = signal[i]*m_table->Lookup(i);
  }

  double tmp1, tmp2;
  for(int i = 0; i < m_halfsize; i++){
    tmp1 = m_diffsig[i+m_halfsize];
    tmp2 = m_diffsig[i];
    m_diffsig[i] = tmp1;
    m_diffsig[i+m_halfsize] = tmp2;

    tmp1 = signal[i+m_halfsize];
    tmp2 = signal[i];
    signal[i] = tmp1;
    signal[i+m_halfsize] = tmp2;
  }

  memcpy(m_fftIn, &signal[0], sizeof(double) * m_fftsize);
  fftw_execute(m_plan);
  fftw_execute(m_diffplan);

  m_output[0] = m_fftOut[0][0] / m_norm;
  m_output[1] = m_fftOut[0][1] / m_norm;

  int i = 2;
  for(int bin = 1; bin < m_halfsize; bin++){
    a = (m_fftOut[bin][0] * 2) / m_norm;
    b = (m_fftOut[bin][1] * 2) / m_norm;
    da = (m_fftdiff[bin][0] * 2) / m_norm;
    db = (m_fftdiff[bin][1] * 2) / m_norm;
    powerspec = (a * a) + (b * b);

    if((m_output[i] = (double)sqrt(powerspec)) != 0.f){
      m_output[i+1] = ((a*db - b*da)/powerspec)*m_factor + bin*m_fund;
      ph = (double) atan2(b, a);
      d = ph - m_phases[bin];
      while(d > PI) d -= TWOPI;
      while(d < -PI) d += TWOPI;
      m_phases[bin] += d;
    }
    else{
      m_output[i+1] = bin*m_fund;
      m_phases[bin] = 0.f ;
    }
    i += 2;
  }
}

short
IFGram::DoProcess(){
  if(!m_error){
    if(m_input){
      if(m_enable){
        double sig = 0.f;

        for(m_vecpos = 0; m_vecpos < m_hopsize; m_vecpos++){
          // signal input
          sig = m_input->Output(m_vecpos);

          // distribute to the signal input frames
          // according to a time pointer (kept by counter[n])
          for(int i = 0; i < m_frames; i++){
            m_sigframe[i][m_counter[i]] = (double) sig;
            m_counter[i]++;
          }
        }

        // every vecsize samples
        // set the current fftframe to be transformed
        m_cur--;
        if(m_cur < 0) m_cur = m_frames - 1;

        // instant frequency analysis
        IFAnalysis(m_sigframe[m_cur]);

        // zero the current fftframe time pointer
        m_counter[m_cur] = 0;
        return 1;
      }
      else{ // if disabled, reset the fftframes
        for(m_vecpos = 0; m_vecpos < m_hopsize; m_vecpos++)
          m_output[m_vecpos] = 0.f;
        return 1;
      }
    }
    else{
      m_error = 3;
      return 0;
    }
  }
  else
    return 0;
}
