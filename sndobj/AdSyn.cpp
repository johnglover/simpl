
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

#include "AdSyn.h"

AdSyn::AdSyn(){
}

AdSyn::AdSyn(SinAnal* input, int maxtracks, Table* table,
             double pitch, double scale, int vecsize, double sr)      
      :ReSyn(input, maxtracks, table, pitch, scale, 1.f, vecsize, sr){
}

AdSyn::~AdSyn(){
}

short
AdSyn::DoProcess(){
    if(m_input){
        double ampnext,amp,freq,freqnext,phase;
        int i3, i, j, ID, track;
        int notcontin = 0;
        bool contin = false;
        int oldtracks = m_tracks;
        double* tab = m_ptable->GetTable(); 
        if((m_tracks = ((SinAnal *)m_input)->GetTracks()) > m_maxtracks){
            m_tracks = m_maxtracks;
        }
        memset(m_output, 0, sizeof(double)*m_vecsize);

        // for each track
        i = j = 0;
        while(i < m_tracks*3){
            i3 = i/3;
            ampnext =  m_input->Output(i)*m_scale;
            freqnext = m_input->Output(i+1)*m_pitch; 
            ID = ((SinAnal *)m_input)->GetTrackID(i3);

            j = i3+notcontin;    

            if(i3 < oldtracks-notcontin){
                if(m_trackID[j]==ID){   
                    // if this is a continuing track      
                    track = j;
                    contin = true;    
                    freq = m_freqs[track];
                    phase = m_phases[track];
                    amp = m_amps[track];
                }
                else{
                    // if this is  a dead track
                    contin = false;
                    track = j;
                    freqnext = freq = m_freqs[track];
                    phase = m_phases[track];
                    amp = m_amps[track]; 
                    ampnext = 0.f;
                }
            }         
            else{ 
                // new tracks
                contin = true;
                track = -1;
                freq = freqnext;
                phase = -freq*m_factor;
                amp = 0.f;
            }

            // interpolation & track synthesis loop
            double a,f,frac,incra,incrph;
            int ndx;
            a = amp;
            f = freq;
            incra = (ampnext - amp)/m_vecsize;
            incrph = (freqnext - freq)/m_vecsize;
            for(m_vecpos=0; m_vecpos < m_vecsize; m_vecpos++){
                if(m_enable) {    
                    // table lookup oscillator
                    phase += f*m_ratio;
                    while(phase < 0) phase += m_size;
                    while(phase >= m_size) phase -= m_size;
                    ndx = Ftoi(phase);
                    frac = phase -  ndx;
                    m_output[m_vecpos] += a*(tab[ndx] + (tab[ndx+1] - tab[ndx])*frac);
                    a += incra;
                    f += incrph;
                }
                else m_output[m_vecpos] = 0.f;      
            }

            // keep amp, freq, and phase values for next time
            if(contin){
                m_amps[i3] = ampnext;
                m_freqs[i3] = freqnext;
                m_phases[i3] = phase;
                m_trackID[i3] = ID;    
                i += 3;
            } 
            else notcontin++;
        } 
        return 1;
    }
    else{
        m_error  = 1;
        return 0;
    }
}
