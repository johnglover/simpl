 
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

#include "IFAdd.h"

IFAdd::IFAdd(){
}

IFAdd::IFAdd(IFGram* input, int bins, Table* table,
	     float pitch, float scale, float tscal, int vecsize, float sr)	  
  : ReSyn((SinAnal *)input, bins, table, pitch, scale, tscal, vecsize, sr){
}

IFAdd::~IFAdd(){
}

short
IFAdd::DoProcess() {
	
  if(m_input){
		
    float ampnext,amp,freq, freqnext, phase;
    float inc1, inc2, a, ph, cnt, frac;
    float a2, a3, phasediff, phasenext, cph, shf;
    bool lock;
    int i2, i, bins = m_maxtracks, ndx;
    float* tab = m_ptable->GetTable(); 
    memset(m_output, 0, sizeof(float)*m_vecsize);

    shf = m_tscal*m_pitch;
    if(shf  - Ftoi(shf)) lock = false;
    else lock = true;
		
    // for each bin from 1
    for(i=1; i < bins; i++){
			
      i2 = i<<1;

      ampnext =  m_input->Output(i2)*m_scale;
      freqnext = m_input->Output(i2+1)*TWOPI*m_pitch;
      phasenext = ((IFGram *)m_input)->Outphases(i)*shf;
      freq = m_freqs[i];
      phase = m_phases[i];
      amp = m_amps[i];
			 
      //phase difference
      phasediff = phasenext - phase;
      while(phasediff >= PI) phasediff -= TWOPI;
      while(phasediff < -PI) phasediff += TWOPI;
      // update phasediff to match the freq
      cph = ((freq+freqnext)*m_factor/2. - phasediff)/TWOPI;
      phasediff += TWOPI* (lock ? Ftoi(cph + 0.5) : cph);
       
      // interpolation coefs
      a2 = 3./m_facsqr * (phasediff - m_factor/3.*(2*freq+freqnext));
      a3 = 1./(3*m_facsqr)  * (freqnext - freq - 2*a2*m_factor);

      // interpolation resynthesis loop	
      a = amp;
      ph = phase;
      cnt = 0;
      inc1 = (ampnext - amp)/m_vecsize;
      inc2 = 1/m_sr;  
      for(m_vecpos=0; m_vecpos < m_vecsize; m_vecpos++){
		
	if(m_enable) {    
	  // table lookup oscillator
	  ph *= m_LoTWOPI;
	  while(ph < 0) ph += m_size;
	  while(ph >= m_size) ph -= m_size;
	  ndx = Ftoi(ph);
	  frac = ph - ndx;
	  m_output[m_vecpos] += a*(tab[ndx] + (tab[ndx+1] - tab[ndx])*frac); 
	  a += inc1;
	  cnt += inc2;
	  ph = phase + cnt*(freq + cnt*(a2 + a3*cnt));
					
	}
	else m_output[m_vecpos] = 0.f;		
      }

      // keep amp, freq, and update phase for next time
      m_amps[i] = ampnext;
      m_freqs[i] = freqnext;
      phasenext += (lock ? 0 : (cph - Ftoi(cph))*TWOPI); 
      while(phasenext < 0) phasenext += TWOPI;
      while(phasenext >= TWOPI) phasenext -= TWOPI;
      m_phases[i] = phasenext;  
    } 
    return 1;
  }
  else {
    m_error  = 1;
    return 0;
  }

}
