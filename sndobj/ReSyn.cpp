 
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

#include "ReSyn.h"
#include "IFGram.h"

ReSyn::ReSyn(){
  AddMsg("pitch", 31);
  AddMsg("timescale", 32);

}

ReSyn::ReSyn(SinAnal* input, int maxtracks, Table* table, float pitch, float scale, float tscal, 
	     int vecsize, float sr)	  
  : SinSyn(input, maxtracks, table, scale, vecsize, sr){
  m_pitch = pitch;
  AddMsg("pitch", 31);
  AddMsg("timescale", 32);
}

ReSyn::~ReSyn(){
}

int
ReSyn::Set(char* mess, float value){

  switch(FindMsg(mess)){

  case 31:
    SetPitch(value);
    return 1;
	
  case 32:
    SetTimeScale(value);
    return 1;
	

  default:
    return SinSyn::Set(mess, value);

  }
}


short
ReSyn::DoProcess() {
	
  if(m_input){
		
    float ampnext,amp,freq, freqnext, phase,phasenext;
    float a2, a3, phasediff, cph;
    int i3, i, j, ID;
    int notcontin = 0;
    bool contin = false;
    int oldtracks = m_tracks;
    float* tab = m_ptable->GetTable(); 
    if((m_tracks = ((SinAnal *)m_input)->GetTracks()) >
       m_maxtracks) m_tracks = m_maxtracks;
		
    memset(m_output, 0, sizeof(float)*m_vecsize);
		
    // for each track
    i = j = 0;
    while(i < m_tracks*3){
			
      i3 = i/3;
      ampnext =  m_input->Output(i)*m_scale;
      freqnext = m_input->Output(i+1)*TWOPI*m_pitch; 
      phasenext = m_input->Output(i+2)*m_tscal*m_pitch;
      ID = ((SinAnal *)m_input)->GetTrackID(i3);

      j = i3+notcontin;
			
      if(i3 < oldtracks-notcontin){
	if(m_trackID[j]==ID){	
	  // if this is a continuing track  	
	  contin = true;	
	  freq = m_freqs[j];
	  phase = m_phases[j];
	  amp = m_amps[j];
					
	}
	else {
	  // if this is  a dead track
	  contin = false;
	  freqnext = freq = m_freqs[j];
	  phase = m_phases[j];
	  phasenext = phase + freq*m_factor;
	  amp = m_amps[j]; 
	  ampnext = 0.f;
	}
      }
			
      else{ 
	// new tracks
	contin = true;
	freq = freqnext;
	phase = phasenext - freq*m_factor;
	amp = 0.f;
      }
			
      //phase difference
      phasediff = phasenext - phase;
      while(phasediff >= PI) phasediff -= TWOPI;
      while(phasediff < -PI) phasediff += TWOPI;
      // update phasediff to match the freq
      cph = ((freq+freqnext)*m_factor/2. - phasediff)/TWOPI;
      phasediff += TWOPI*cph;
		   
      // interpolation coefs
      a2 = 3./m_facsqr * (phasediff - m_factor/3.*(2*freq+freqnext));
      a3 = 1./(3*m_facsqr)  * (freqnext - freq - 2*a2*m_factor);

      // interpolation resynthesis loop	
      float inc1, inc2, a, ph, cnt, frac;
      int ndx;
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

      // keep amp, freq, and phase values for next time
      if(contin){
	m_amps[i3] = ampnext;
	m_freqs[i3] = freqnext;
	phasenext += (cph - Ftoi(cph))*TWOPI; 
	while(phasenext < 0) phasenext += TWOPI;
	while(phasenext >= TWOPI) phasenext -= TWOPI;
	m_phases[i3] = phasenext;  
	m_trackID[i3] = ID;    
	i += 3;
      } else notcontin++;
    } 
    return 1;
  }
  else {
    m_error  = 1;
    return 0;
  }
 
}
