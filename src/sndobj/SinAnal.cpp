 
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

#include "SinAnal.h"

SinAnal::SinAnal(){
	m_thresh = 0.f;
	m_startupThresh = 0.f;
	m_maxtracks = 0;
	m_tracks = 0;
	m_prev = 1; m_cur =0;
	m_numbins = m_accum = 0;
	
	m_bndx = m_pkmags = m_adthresh = 0;
	m_phases = m_freqs = m_mags = m_bins = 0;
	m_tstart = m_lastpk = m_trkid = 0;
	m_trndx = 0;
	m_binmax = m_magmax = m_diffs = 0;
	m_maxix = 0;
	m_contflag = 0;
	m_minpoints = 0;
	m_maxgap = 3;
	m_numpeaks = 0;
	AddMsg("max tracks", 21);
	AddMsg("threshold", 22);
}

SinAnal::SinAnal(SndObj* input, double threshold, int maxtracks, 
				 int minpoints, int maxgap, double sr)
				 :SndObj(input,maxtracks*3,sr){
	m_minpoints = (minpoints > 1 ? minpoints : 1) - 1;
	m_thresh = threshold;
	m_startupThresh = 0.f;
	m_maxtracks = maxtracks;
	m_tracks = 0;
	m_prev = 1; m_cur = 0; m_accum = 0;
	m_maxgap = maxgap;
	m_numpeaks = 0;
	m_numbins = ((FFT *)m_input)->GetFFTSize()/2 + 1;
	
	m_bndx = new double*[m_minpoints+2];
	m_pkmags = new double*[m_minpoints+2];
	m_adthresh = new double*[m_minpoints+2];
	m_tstart = new unsigned int*[m_minpoints+2];
	m_lastpk = new unsigned int*[m_minpoints+2];
	m_trkid = new unsigned int*[m_minpoints+2];
	int i;
	for(i = 0; i < m_minpoints+2; i++){
		m_bndx[i] = new double[m_maxtracks];
		memset(m_bndx[i], 0, sizeof(double) * m_maxtracks);
		m_pkmags[i] = new double[m_maxtracks];
		memset(m_pkmags[i], 0, sizeof(double) * m_maxtracks);
		m_adthresh[i] = new double[m_maxtracks];
		memset(m_adthresh[i], 0, sizeof(double) * m_maxtracks);
		m_tstart[i] = new unsigned int[m_maxtracks];
        memset(m_tstart[i], 0, sizeof(unsigned int) * m_maxtracks);
		m_lastpk[i] = new unsigned int[m_maxtracks];
        memset(m_lastpk[i], 0, sizeof(unsigned int) * m_maxtracks);
		m_trkid[i] = new unsigned int[m_maxtracks];
        memset(m_trkid[i], 0, sizeof(unsigned int) * m_maxtracks);
	}
	
	m_bins = new double[m_maxtracks];
	memset(m_bins, 0, sizeof(double) * m_maxtracks);
	m_trndx = new int[m_maxtracks];
	memset(m_trndx, 0, sizeof(int) * m_maxtracks);
	m_contflag = new bool[m_maxtracks];
	memset(m_contflag, 0, sizeof(bool) * m_maxtracks);

	m_phases = new double[m_numbins];
	memset(m_phases, 0, sizeof(double) * m_numbins);
	m_freqs = new double[m_numbins];
	memset(m_freqs, 0, sizeof(double) * m_numbins);
	m_mags = new double[m_numbins];
	memset(m_mags, 0, sizeof(double) * m_numbins);

	m_binmax = new double[m_numbins];
	memset(m_binmax, 0, sizeof(double) * m_numbins);
	m_magmax = new double[m_numbins];
	memset(m_magmax, 0, sizeof(double) * m_numbins);
	m_diffs = new double[m_numbins];
	memset(m_diffs, 0, sizeof(double) * m_numbins);
	
	m_maxix = new int[m_numbins];
	memset(m_maxix, 0, sizeof(int) * m_numbins);
	m_timecount = 0;
	
	m_phases[0] = 0.f;
	m_freqs[0] = 0.f;
	m_phases[m_numbins-1] = 0.f;
	m_freqs[m_numbins-1] = m_sr/2;
		
	AddMsg("max tracks", 21);
	AddMsg("threshold", 22);
	
	for(i = 0; i < m_maxtracks; i++) {
		m_pkmags[0][i] = m_bndx[0][i] = m_adthresh[0][i] = 0.f;
		m_pkmags[1][i] = m_bndx[1][i] = m_adthresh[1][i] = 0.f;
    }
}

SinAnal::SinAnal(SndObj* input, int numbins, double threshold, int maxtracks,
				 int minpoints, int maxgap, double sr)
				 :SndObj(input,maxtracks*3,sr){
	m_minpoints = (minpoints > 1 ? minpoints : 1) - 1;
	m_thresh = threshold;
	m_startupThresh = 0.f;
	m_maxtracks = maxtracks;
	m_tracks = 0;
	m_prev = 1; m_cur = 0; m_accum = 0;
	m_maxgap = maxgap;
	m_numpeaks = 0;
	m_numbins = numbins;

	m_bndx = new double*[m_minpoints+2];
	m_pkmags = new double*[m_minpoints+2];
	m_adthresh = new double*[m_minpoints+2];
	m_tstart = new unsigned int*[m_minpoints+2];
	m_lastpk = new unsigned int*[m_minpoints+2];
	m_trkid = new unsigned int*[m_minpoints+2];
	int i;
	for(i = 0; i < m_minpoints+2; i++){
		m_bndx[i] = new double[m_maxtracks];
		memset(m_bndx[i], 0, sizeof(double) * m_maxtracks);
		m_pkmags[i] = new double[m_maxtracks];
		memset(m_pkmags[i], 0, sizeof(double) * m_maxtracks);
		m_adthresh[i] = new double[m_maxtracks];
		memset(m_adthresh[i], 0, sizeof(double) * m_maxtracks);
		m_tstart[i] = new unsigned int[m_maxtracks];
        memset(m_tstart[i], 0, sizeof(unsigned int) * m_maxtracks);
		m_lastpk[i] = new unsigned int[m_maxtracks];
        memset(m_lastpk[i], 0, sizeof(unsigned int) * m_maxtracks);
		m_trkid[i] = new unsigned int[m_maxtracks];
        memset(m_trkid[i], 0, sizeof(unsigned int) * m_maxtracks);
	}

	m_bins = new double[m_maxtracks];
	memset(m_bins, 0, sizeof(double) * m_maxtracks);
	m_trndx = new int[m_maxtracks];
	memset(m_trndx, 0, sizeof(int) * m_maxtracks);
	m_contflag = new bool[m_maxtracks];
	memset(m_contflag, 0, sizeof(bool) * m_maxtracks);

	m_phases = new double[m_numbins];
	memset(m_phases, 0, sizeof(double) * m_numbins);
	m_freqs = new double[m_numbins];
	memset(m_freqs, 0, sizeof(double) * m_numbins);
	m_mags = new double[m_numbins];
	memset(m_mags, 0, sizeof(double) * m_numbins);

	m_binmax = new double[m_numbins];
	memset(m_binmax, 0, sizeof(double) * m_numbins);
	m_magmax = new double[m_numbins];
	memset(m_magmax, 0, sizeof(double) * m_numbins);
	m_diffs = new double[m_numbins];
	memset(m_diffs, 0, sizeof(double) * m_numbins);

	m_maxix = new int[m_numbins];
	memset(m_maxix, 0, sizeof(int) * m_numbins);
	m_timecount = 0;

	m_phases[0] = 0.f;
	m_freqs[0] = 0.f;
	m_phases[m_numbins-1] = 0.f;
	m_freqs[m_numbins-1] = m_sr/2;

	AddMsg("max tracks", 21);
	AddMsg("threshold", 22);

	for(i = 0; i < m_maxtracks; i++) {
		m_pkmags[0][i] = m_bndx[0][i] = m_adthresh[0][i] = 0.f;
		m_pkmags[1][i] = m_bndx[1][i] = m_adthresh[1][i] = 0.f;
    }
}

SinAnal::~SinAnal(){
    if(m_numbins){
        int i;
        for(i = 0; i < m_minpoints+2; i++){
            delete [] m_bndx[i];
            delete [] m_pkmags[i];
            delete [] m_adthresh[i];
            delete [] m_tstart[i];
            delete [] m_lastpk[i];
            delete [] m_trkid[i];
        }
    }

	if(m_bndx) delete[] m_bndx;  
	if(m_pkmags) delete[] m_pkmags;  
	if(m_adthresh) delete[] m_adthresh; 
	if(m_tstart) delete[] m_tstart; 
	if(m_lastpk) delete[] m_lastpk; 
	if(m_trkid) delete[] m_trkid;  
	if(m_bins) delete[] m_bins;
	if(m_trndx) delete[] m_trndx;
	if(m_contflag) delete[] m_contflag;

	if(m_phases) delete[] m_phases;
	if(m_freqs) delete[] m_freqs;
	if(m_mags) delete[] m_mags;
	if(m_binmax) delete[] m_binmax;
	if(m_magmax) delete[] m_magmax;
	if(m_diffs) delete[] m_diffs;
	if(m_maxix) delete[] m_maxix;
}

void
SinAnal::SetMaxTracks(int maxtracks){
    if(m_numbins){
        int i;
        for(i = 0; i < m_minpoints+2; i++){
            delete [] m_bndx[i];
            delete [] m_pkmags[i];
            delete [] m_adthresh[i];
            delete [] m_tstart[i];
            delete [] m_lastpk[i];
            delete [] m_trkid[i];
        }
    }

	if(m_bndx) delete[] m_bndx;  
	if(m_pkmags) delete[] m_pkmags;  
	if(m_adthresh) delete[] m_adthresh; 
	if(m_tstart) delete[] m_tstart; 
	if(m_lastpk) delete[] m_lastpk; 
	if(m_trkid) delete[] m_trkid;  
	if(m_bins) delete[] m_bins;
	if(m_trndx) delete[] m_trndx;
	if(m_contflag) delete[] m_contflag;

	m_maxtracks = maxtracks;
	m_bins = new double[m_maxtracks];
	memset(m_bins, 0, sizeof(double) * m_maxtracks);
	m_trndx = new int[m_maxtracks];
	memset(m_trndx, 0, sizeof(int) * m_maxtracks);
	m_contflag = new bool[m_maxtracks];
	memset(m_contflag, 0, sizeof(bool) * m_maxtracks);

	m_prev = 1; 
    m_cur = 0;
	m_bndx = new double*[m_minpoints+2];
	m_pkmags = new double*[m_minpoints+2];
	m_adthresh = new double*[m_minpoints+2];
	m_tstart = new unsigned int*[m_minpoints+2];
	m_lastpk = new unsigned int*[m_minpoints+2];
	m_trkid = new unsigned int*[m_minpoints+2];
	int i;
	for(i = 0; i < m_minpoints+2; i++){
        m_bndx[i] = new double[m_maxtracks];
		memset(m_bndx[i], 0, sizeof(double) * m_maxtracks);
		m_pkmags[i] = new double[m_maxtracks];
		memset(m_pkmags[i], 0, sizeof(double) * m_maxtracks);
		m_adthresh[i] = new double[m_maxtracks];
		memset(m_adthresh[i], 0, sizeof(double) * m_maxtracks);
		m_tstart[i] = new unsigned int[m_maxtracks];
        memset(m_tstart[i], 0, sizeof(unsigned int) * m_maxtracks);
		m_lastpk[i] = new unsigned int[m_maxtracks];
        memset(m_lastpk[i], 0, sizeof(unsigned int) * m_maxtracks);
		m_trkid[i] = new unsigned int[m_maxtracks];
        memset(m_trkid[i], 0, sizeof(unsigned int) * m_maxtracks);
	}
	for(i = 0; i < m_maxtracks; i++)
		m_pkmags[m_prev][i] = m_bndx[m_prev][i] = m_adthresh[m_prev][i] = 0.f;
	
	SetVectorSize(m_maxtracks*3);
}

void
SinAnal::SetIFGram(SndObj* input){
	if(m_input){
		delete[] m_phases;
		delete[] m_freqs;
		delete[] m_mags;
		delete[] m_binmax;
		delete[] m_magmax;
		delete[] m_diffs;
		delete[] m_maxix;
	}
	
	SetInput(input);
	m_numbins = ((FFT *)m_input)->GetFFTSize()/2 + 1;
	
	m_phases = new double[m_numbins];
	m_freqs = new double[m_numbins];
	m_mags = new double[m_numbins];
	m_binmax = new double[m_numbins];
	m_magmax = new double[m_numbins];
	m_diffs = new double[m_numbins];
	m_maxix = new int[m_numbins];
	
	m_phases[0] = 0.f;
	m_freqs[0] = 0.f;
	m_phases[m_numbins-1] = 0.f;
	m_freqs[m_numbins-1] = m_sr/2;
}

int
SinAnal::Set(const char* mess, double value){
	switch(FindMsg(mess)){
        case 21:
            SetMaxTracks((int)value);
            return 1;
            
        case 22:
            SetThreshold(value);
            return 1;
            
        default:
            return	SndObj::Set(mess, value);
	}
}

int
SinAnal::Connect(const char* mess, void *input){
	switch(FindMsg(mess)){
        case 3:
            SetIFGram((SndObj *)input);
            return 1;

        default:
            return SndObj::Connect(mess, input);
	}
}

int
SinAnal::peakdetection(){
	double logthresh;
	int i = 0, n = 0;
	double max = 0.f;
	double y1, y2, a, b, ftmp;
	
	for(i=0; i<m_numbins;i++)
		if(max < m_mags[i]) max = m_mags[i];
		
	m_startupThresh = m_thresh*max;
	logthresh = log(m_startupThresh/5.f);
		
	// Quadratic Interpolation 
	// obtains bin indexes and magnitudes
	// m_binmax & m_magmax respectively
	
	bool test1 = true, test2 = false;
	
	// take the logarithm of the magnitudes
	for(i=0; i<m_numbins;i++) {
        if(m_mags[i] > 0) {
            m_mags[i] = log(m_mags[i]);
        }
    }
	
	for(i=0;i < m_numbins-1; i++) {
		
		if(i) test1 = (m_mags[i] > m_mags[i-1] ? true : false );
		else test1 = false;
		test2 = (m_mags[i] >= m_mags[i+1] ? true : false); // check!
		
		if((m_mags[i] > logthresh) && 
			(test1 && test2)){
			m_maxix[n] = i;
			n++;
		}
	}

	for(i = 0; i < n; i++){
		int rmax;
		rmax = m_maxix[i];
		
		y1 = m_mags[rmax] - (ftmp = (rmax ? m_mags[rmax-1] : m_mags[rmax+1])) + 0.000001;
		y2 = (rmax < m_numbins-1 ? m_mags[rmax+1] : m_mags[rmax]) - ftmp + 0.000001;
		
		a = (y2 - 2*y1)/2.f;
		b = 1.f - y1/a;
		
		m_binmax[i] = (double) (rmax - 1. + b/2.);
		m_magmax[i] = (double) exp(ftmp - a*b*b/4.);
	}
	
	return n;
}

int
SinAnal::FindPeaks(){
	if(!m_error){
		if(m_input){
			int i2;
			// input is in "real-spec" format packing 0 and Nyquist
			// together in pos 0 and 1
			for(m_vecpos=1; m_vecpos < m_numbins-1; m_vecpos++){
				i2 = m_vecpos*2;
				m_phases[m_vecpos] = ((PVA *)m_input)->Outphases(m_vecpos);
				m_freqs[m_vecpos] = m_input->Output(i2+1);
				m_mags[m_vecpos] = m_input->Output(i2);
			}
			m_mags[0] = m_input->Output(0);
			m_mags[m_numbins-1] = m_input->Output(1);

			if(m_enable){
				// find peaks
				int n = peakdetection();

				// output peaks
				for(m_vecpos=0; m_vecpos < m_vecsize; m_vecpos += 3){
					int pos = m_vecpos/3, ndx;
					double frac, a, b;
					if((pos < n) && (pos < m_maxtracks)){
						// bin number
						ndx = Ftoi(m_binmax[pos]);
						// fractional part of bin number
						frac = (m_binmax[pos] - ndx);

						// amplitude
						m_output[m_vecpos] = m_magmax[pos];
						// frequency
						a = m_freqs[ndx];
						b = (m_binmax[pos] < m_numbins-1 ? (m_freqs[ndx+1] - a) : 0);
						m_output[m_vecpos+1] = a + frac*b;
						// phase
						m_output[m_vecpos+2] = m_phases[ndx];
					}
					else{
						m_output[m_vecpos] =
						m_output[m_vecpos+1] =
						m_output[m_vecpos+2] = 0.f;
					}
				}
				if(n > m_maxtracks){
					n = m_maxtracks;
				}
				return n;
			}
			else // if disabled
				for(m_vecpos=0; m_vecpos < m_vecsize;m_vecpos++)
					m_output[m_vecpos] = 0.f;
				return 1;
		}
		else {
			m_error = 11;
			return 0;
		}
	}
	return 0;
}

void
SinAnal::SetPeaks(int numamps, double* amps, int numfreqs,
		          double* freqs, int numphases, double* phases){
	double binwidth = (m_sr / 2) / m_numbins;
	m_numpeaks = numamps;
	for(int i = 0; i < m_numbins; i++){
		if(i < m_numpeaks){
			m_magmax[i] = amps[i];
			m_binmax[i] = freqs[i] / binwidth;
			m_phases[i] = phases[i];
		}
		else{
			m_magmax[i] = m_binmax[i] = m_phases[i] = 0.f;
		}
	}
}

void
SinAnal::PartialTracking(){
	int bestix, count=0, i = 0, n = 0, j = 0;
	double dbstep;

	// reset allowcont flags
	for(i=0; i < m_maxtracks; i++){
		m_contflag[i] = false;
	}

	// loop to the end of tracks (indicate by the 0'd bins)
	// find continuation tracks
	for(j=0; j < m_maxtracks && m_bndx[m_prev][j] != 0.f; j++){
		int foundcont = 0;

		if(m_numpeaks > 0){ // check for peaks; m_numpeaks will be > 0
			double F = m_bndx[m_prev][j];

			for(i=0; i < m_numbins; i++){
				m_diffs[i] = m_binmax[i] - F; //differences
				m_diffs[i] = (m_diffs[i] < 0 ? -m_diffs[i] : m_diffs[i]);
			}

			bestix = 0;  // best index
			for(i=0; i < m_numbins; i++)
				if(m_diffs[i] < m_diffs[bestix]) bestix = i;

            // if difference smaller than 1 bin
            double tempf = F -  m_binmax[bestix];
            tempf = (tempf < 0 ? -tempf : tempf);
            if(tempf < 1.){
                // if amp jump is too great (check)
                if(m_adthresh[m_prev][j] <
                    (dbstep = 20*log10(m_magmax[bestix]/m_pkmags[m_prev][j]))){
                    // mark for discontinuation;
                    m_contflag[j] = false;
                }
                else{
                    m_bndx[m_prev][j] = m_binmax[bestix];
                    m_pkmags[m_prev][j] = m_magmax[bestix];
                    // track index keeps track history
                    // so we know which ones continue
                    m_contflag[j] = true;
                    m_binmax[bestix] = m_magmax[bestix] = 0.f;
                    m_lastpk[m_prev][j] = m_timecount;
                    foundcont = 1;
                    count++;

                    // update the adaptive mag threshold
                    double tmp1 = dbstep*1.5f;
                    double tmp2 = m_adthresh[m_prev][j] -
                        (m_adthresh[m_prev][j] - 1.5f)*0.048770575f;
                    m_adthresh[m_prev][j] = (tmp1 > tmp2 ? tmp1 : tmp2);
                }  // else
            } // if difference
		} // if check

		// if we did not find a continuation
		// we'll check if the magnitudes around it are below
		// a certain threshold. Mags[] holds the logs of the magnitudes
		// Check also if the last peak in this track is more than m_maxgap
		// old
		if(!foundcont){
			if((exp(m_mags[int(m_bndx[m_prev][j]+0.5)]) < 0.2*m_pkmags[m_prev][j])
				|| ((m_timecount - m_lastpk[m_prev][j]) > (unsigned int) m_maxgap)){
				m_contflag[j] = false;
			} 
            else{
                m_contflag[j] = true;
                count++;
			}
		}
	} // for loop

	// compress the arrays
	for(i=0, n=0; i < m_maxtracks; i++){
		if(m_contflag[i]){
			m_bndx[m_cur][n] = m_bndx[m_prev][i];
			m_pkmags[m_cur][n] = m_pkmags[m_prev][i];
			m_adthresh[m_cur][n] = m_adthresh[m_prev][i];
			m_tstart[m_cur][n] = m_tstart[m_prev][i];
			m_trkid[m_cur][n] = m_trkid[m_prev][i];
			m_lastpk[m_cur][n] = m_lastpk[m_prev][i];
			n++;
		}	// ID == -1 means zero'd track
		else
			m_trndx[i] = -1;
	}

	if(count < m_maxtracks){
		// if we have not exceeded available tracks.
		// create new tracks for all new peaks
		for(j=0; j< m_numbins && count < m_maxtracks; j++){
			if(m_magmax[j] > m_startupThresh){
                m_bndx[m_cur][count] = m_binmax[j];
                m_pkmags[m_cur][count] = m_magmax[j];
                m_adthresh[m_cur][count] = 400.f;
				// track ID is a positive number in the
				// range of 0 - maxtracks*3 - 1
				// it is given when the track starts
				// used to identify and match tracks
				m_tstart[m_cur][count] = m_timecount;
				m_trkid[m_cur][count] = ((m_accum++)%m_vecsize);
				m_lastpk[m_cur][count] = m_timecount;
				count++;
			}
		}
		for(i = count; i < m_maxtracks; i++){
			// zero the right-hand size of the current arrays
			if(i >= count)
				m_pkmags[m_cur][i] = m_bndx[m_cur][i] = m_adthresh[m_cur][i] = 0.f;
		}
	} // if count != maxtracks

	// count is the number of continuing tracks + new tracks
	// now we check for tracks that have been there for more
	// than minpoints hop periods and output them
	m_tracks = 0;
	for(i=0; i < count; i++){
		int curpos = m_timecount-m_minpoints;
		if(curpos >= 0 && m_tstart[m_cur][i] <= (unsigned int)curpos){
			int tpoint = m_cur-m_minpoints;
			if(tpoint < 0){
				tpoint += m_minpoints+2;
			}

			m_bins[i] = m_bndx[tpoint][i];
			m_mags[i] = m_pkmags[tpoint][i];
			m_trndx[i] = m_trkid[tpoint][i];
			m_tracks++;
		}
	}

	// end track-selecting
	// current arrays become previous
    m_prev = m_cur;
    m_cur = (m_cur < m_minpoints+1 ? m_cur+1 : 0);
	m_timecount++;

	// Output
	if(!m_error){
		if(m_input){
			if(m_enable){
				double binwidth = (m_sr / 2) / m_numbins;

				for(m_vecpos=0; m_vecpos < m_vecsize; m_vecpos += 3){
					int pos = m_vecpos/3;
					if((pos < m_tracks)  && (pos < m_maxtracks)){
						// amplitude
						m_output[m_vecpos] = m_mags[pos];
						// frequency
						m_output[m_vecpos+1] = m_bins[pos] * binwidth;
						// phase
						m_output[m_vecpos+2] = m_phases[pos];
					}
					else{
						m_output[m_vecpos] =
						m_output[m_vecpos+1] =
						m_output[m_vecpos+2] = 0.f;
					}
				}
			}
			else // if disabled
				for(m_vecpos=0; m_vecpos < m_vecsize;m_vecpos++)
					m_output[m_vecpos] = 0.f;
		}
		else {
			m_error = 11;
		}
	}
}

void
SinAnal::sinanalysis(){

	int bestix, count=0, i = 0, n = 0, j = 0;
	double dbstep;

	n = peakdetection();
		
	// track-secting
	
	// reset allowcont flags 
	for(i=0; i<m_maxtracks;i++){
		m_contflag[i] = false;
	}
	
	// loop to the end of tracks (indicate by the 0'd bins)
	// find continuation tracks
	
	for(j=0; m_bndx[m_prev][j] != 0.f && j < m_maxtracks; j++){
		
		int foundcont = 0;
		
		if(n > 0){ // check for peaks; n will be > 0
			
			double F = m_bndx[m_prev][j];
			
			for(i=0; i < m_numbins; i++){
				m_diffs[i] = m_binmax[i] - F; //differences
				m_diffs[i] = (m_diffs[i] < 0 ? -m_diffs[i] : m_diffs[i]);
			}
			
			
			bestix = 0;  // best index
			for(i=0; i < m_numbins; i++) 
				if(m_diffs[i] < m_diffs[bestix]) bestix = i;
				
				// if difference smaller than 1 bin
				double tempf = F -  m_binmax[bestix];
				tempf = (tempf < 0 ? -tempf : tempf);
				if(tempf < 1.){
					
					// if amp jump is too great (check)
					if(m_adthresh[m_prev][j] < 
						(dbstep = 20*log10(m_magmax[bestix]/m_pkmags[m_prev][j]))){
						// mark for discontinuation;  
						m_contflag[j] = false;							
					}
					else {
						m_bndx[m_prev][j] = m_binmax[bestix];
						m_pkmags[m_prev][j] = m_magmax[bestix];
						// track index keeps track history
						// so we know which ones continue
						m_contflag[j] = true; 
						m_binmax[bestix] = m_magmax[bestix] = 0.f;
						m_lastpk[m_prev][j] = m_timecount;
						foundcont = 1;
						count++;
						
						// update the adaptive mag threshold 
						double tmp1 = dbstep*1.5f;
						double tmp2 = m_adthresh[m_prev][j] -
							(m_adthresh[m_prev][j] - 1.5f)*0.048770575f;
						m_adthresh[m_prev][j] = (tmp1 > tmp2 ? tmp1 : tmp2);
						
					}  // else  		
				} // if difference          
				// if check
		}
		
		// if we did not find a continuation
		// we'll check if the magnitudes around it are below
		// a certain threshold. Mags[] holds the logs of the magnitudes
		// Check also if the last peak in this track is more than m_maxgap
		// old
		if(!foundcont){ 
			if((exp(m_mags[int(m_bndx[m_prev][j]+0.5)]) < 0.2*m_pkmags[m_prev][j])
				|| ((m_timecount - m_lastpk[m_prev][j]) > (unsigned int) m_maxgap))
			{
				m_contflag[j] = false;

			} else {
                m_contflag[j] = true;
                count++;
			}

		}	
			
	} // for loop 
	
	// compress the arrays
	for(i=0, n=0; i < m_maxtracks; i++){
		if(m_contflag[i]){
			m_bndx[m_cur][n] = m_bndx[m_prev][i];
			m_pkmags[m_cur][n] = m_pkmags[m_prev][i];
			m_adthresh[m_cur][n] = m_adthresh[m_prev][i];
			m_tstart[m_cur][n] = m_tstart[m_prev][i];
			m_trkid[m_cur][n] = m_trkid[m_prev][i];
			m_lastpk[m_cur][n] = m_lastpk[m_prev][i];
			n++;
		}	// ID == -1 means zero'd track
		else
			m_trndx[i] = -1;
	}

	if(count < m_maxtracks){
		// if we have not exceeded available tracks.	
		// create new tracks for all new peaks 
		
		for(j=0; j< m_numbins && count < m_maxtracks; j++){
			
			if(m_magmax[j] > m_startupThresh){
				
				m_bndx[m_cur][count] = m_binmax[j];    
				m_pkmags[m_cur][count] = m_magmax[j];      
				m_adthresh[m_cur][count] = 400.f;    
				// track ID is a positive number in the
				// range of 0 - maxtracks*3 - 1
				// it is given when the track starts
				// used to identify and match tracks
				m_tstart[m_cur][count] = m_timecount;
				m_trkid[m_cur][count] = ((m_accum++)%m_vecsize);
				m_lastpk[m_cur][count] = m_timecount;
				count++;
				
			}
		}		
		for(i = count; i < m_maxtracks; i++){
			// zero the right-hand size of the current arrays
			if(i >= count)
				m_pkmags[m_cur][i] = m_bndx[m_cur][i] = m_adthresh[m_cur][i] = 0.f;
		} 
		
	} // if count != maxtracks
	
	// count is the number of continuing tracks + new tracks
	// now we check for tracks that have been there for more
	// than minpoints hop periods and output them
	
	m_tracks = 0;
	for(i=0; i < count; i++){
		int curpos = m_timecount-m_minpoints;
		if(curpos >= 0 && m_tstart[m_cur][i] <= (unsigned int)curpos){
			int tpoint = m_cur-m_minpoints;
			 
			if(tpoint < 0) {
				tpoint += m_minpoints+2;
			}
			m_bins[i] = m_bndx[tpoint][i];
			m_mags[i] = m_pkmags[tpoint][i];
			m_trndx[i] = m_trkid[tpoint][i];
			m_tracks++;
		}
		
	}
	// end track-selecting
	// current arrays become previous
    //int tmp = m_prev;
	m_prev = m_cur;
	m_cur = (m_cur < m_minpoints+1 ? m_cur+1 : 0);
	m_timecount++;		
}

short
SinAnal::DoProcess(){
	if(!m_error){     
		if(m_input){
			int i2;

			// input is in "real-spec" format packing 0 and Nyquist
			// together in pos 0 and 1

			for(m_vecpos=1; m_vecpos < m_numbins-1; m_vecpos++){
				i2 = m_vecpos*2;
				m_phases[m_vecpos] = ((PVA *)m_input)->Outphases(m_vecpos);
				m_freqs[m_vecpos] = m_input->Output(i2+1);
				m_mags[m_vecpos] = m_input->Output(i2);
			} 
			m_mags[0] = m_input->Output(0);   
			m_mags[m_numbins-1] = m_input->Output(1);
			
			if(m_enable){
				
				// sinusoidal analysis 
				// generates bin indexes and magnitudes
				// m_bins and m_mags, respectively
				
				sinanalysis();
				
				// m_output holds [amp, freq, pha]
				// m_vecsize is m_maxtracks*3 
				// estimated to be a little above count*3
				
				for(m_vecpos=0; m_vecpos < m_vecsize; m_vecpos+=3){
					int pos = m_vecpos/3, ndx;
					double frac,a,b;
					if(pos < m_tracks){
						// magnitudes
						ndx = Ftoi(m_bins[pos]);
						m_output[m_vecpos] = m_mags[pos];
						// fractional part of bin indexes
						frac =(m_bins[pos] - ndx);
						// freq Interpolation
						// m_output[1,4,7, ..etc] holds track freq
						a = m_freqs[ndx];
						b = (m_bins[pos] < m_numbins-1 ? (m_freqs[ndx+1] - a) : 0);
						m_output[m_vecpos+1] = a + frac*b;
						// phase Interpolation
						// m_output[2,5,8 ...] holds track phase
						m_output[m_vecpos+2] = m_phases[ndx];
					}
					else{ // empty tracks
						m_output[m_vecpos] = 
						m_output[m_vecpos+1] = 
						m_output[m_vecpos+2] = 0.f;
					}
				}

								
			}
			else // if disabled
				for(m_vecpos=0; m_vecpos < m_vecsize;m_vecpos++)	  
					m_output[m_vecpos] = 0.f;
				return 1;
		} 
		else {
			m_error = 11;        
			return 0;
		}
	}
	else return 0;
}

