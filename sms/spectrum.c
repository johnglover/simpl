/* 
 * Copyright (c) 2008 MUSIC TECHNOLOGY GROUP (MTG)
 *                         UNIVERSITAT POMPEU FABRA 
 * 
 * 
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 * 
 */
/*! \file spectrum.c
 * \brief functions to convert between frequency (spectrum) and time (wavefrom) domain
 */
#include "sms.h"

/*! \brief compute a complex spectrum from a waveform 
 *              
 * \param sizeWindow size of analysis window
 * \param pWaveform  pointer to input waveform 
 * \param pWindow    pointer to input window
 * \param sizeMag    size of output magnitude and phase spectrums
 * \param pMag       pointer to output magnitude spectrum 
 * \param pPhase     pointer to output phase spectrum 
 * \return sizeFft, -1 on error \todo remove this return
 */
int sms_spectrum(int sizeWindow, sfloat *pWaveform, sfloat *pWindow, int sizeMag, 
                 sfloat *pMag, sfloat *pPhase, sfloat *pFftBuffer)
{
    int i, it2;
    sfloat fReal, fImag;
    
    int sizeFft = sizeMag << 1;
    memset(pFftBuffer, 0, sizeFft * sizeof(sfloat));

    /* apply window to waveform and center window around 0 (zero-phase windowing)*/
    sms_windowCentered(sizeWindow, pWaveform, pWindow, sizeFft, pFftBuffer);
    sms_fft(sizeFft, pFftBuffer);

    /* convert from rectangular to polar coordinates */
    for(i = 0; i < sizeMag; i++)
    { 
        it2 = i << 1; //even numbers 0-N
        fReal = pFftBuffer[it2]; /*odd numbers 1->N+1 */
        fImag = pFftBuffer[it2 + 1]; /*even numbers 2->N+2 */
        pMag[i] = sqrt(fReal * fReal + fImag * fImag);
        pPhase[i] = atan2(-fImag, fReal); /* \todo why is fImag negated? */
        /*pPhase[i] = atan2(fImag, fReal);*/
    }

    return sizeFft;
}

/* sms_spectrum, but without zero-phase windowing, and with phase calculated
 * according by arctan(imag/real) instead of arctan2(-imag/real)
 */
int sms_spectrumW(int sizeWindow, sfloat *pWaveform, sfloat *pWindow, int sizeMag, 
                  sfloat *pMag, sfloat *pPhase, sfloat *pFftBuffer)
{
    int i, it2;
    sfloat fReal, fImag;
    
    int sizeFft = sizeMag << 1;
    memset(pFftBuffer, 0, sizeFft * sizeof(sfloat));

    /* apply window to waveform */
    for(i = 0; i < sizeWindow; i++)
        pFftBuffer[i] = pWaveform[i] * pWindow[i];

    sms_fft(sizeFft, pFftBuffer);

    /* convert from rectangular to polar coordinates */
    for(i = 0; i < sizeMag; i++)
    { 
        it2 = i << 1; //even numbers 0-N
        fReal = pFftBuffer[it2]; /*odd numbers 1->N+1 */
        fImag = pFftBuffer[it2 + 1]; /*even numbers 2->N+2 */
        pMag[i] = sqrt(fReal * fReal + fImag * fImag);
        pPhase[i] = atan2(fImag, fReal);
    }

    return sizeFft;
}

/*! \brief compute the spectrum Magnitude of a waveform
 *
 * This function windows the waveform with pWindow and 
 * performs a zero-padded FFT (if sizeMag*2 > sizeWindow).
 * The spectra is then converted magnitude (RMS).
 *              
 * \param sizeWindow size of analysis window / input wavefrom
 * \param pWaveform  pointer to input waveform
 * \param pWindow    pointer to analysis window 
 * \param sizeMag    size of output magnitude spectrum 
 * \param pMag       pointer to output magnitude spectrum 
 * \return 0 on success, -1 on error
 */
int sms_spectrumMag(int sizeWindow, sfloat *pWaveform, sfloat *pWindow,  
                    int sizeMag, sfloat *pMag, sfloat *pFftBuffer)
{
    int i,it2;
    int sizeFft = sizeMag << 1;
    sfloat fReal, fImag;

    /* apply window to waveform, zero the rest of the array */
    for(i = 0; i < sizeWindow; i++)
        pFftBuffer[i] =  pWaveform[i] * pWindow[i];
    for(i = sizeWindow; i < sizeFft; i++)
        pFftBuffer[i]  = 0.;

    /* compute real FFT */
    sms_fft(sizeFft, pFftBuffer); 

    /* convert from rectangular to polar coordinates */
    for(i = 0; i < sizeMag; i++)
    {
        it2 = i << 1;
        fReal = pFftBuffer[it2];
        fImag = pFftBuffer[it2+1];
        pMag[i] = sqrtf(fReal * fReal + fImag * fImag);
    }

    return sizeFft;
}

/*! \brief function for a quick inverse spectrum, windowed
 *
 *  Not done yet, but this will be a function that is the inverse of
 *  sms_spectrum above.
 *
 * function to perform the inverse FFT, windowing the output
 * sfloat *pFMagSpectrum   input magnitude spectrum
 * sfloat *pFPhaseSpectrum input phase spectrum
 * int sizeFft             size of FFT
 * sfloat *pFWaveform      output waveform
 * int sizeWave            size of output waveform
 * sfloat *pFWindow        synthesis window
 */
int sms_invSpectrum(int sizeWaveform, sfloat *pWaveform, sfloat *pWindow,
                    int sizeMag, sfloat *pMag, sfloat *pPhase, sfloat *pFftBuffer)
{
    int i;
    int sizeFft = sizeMag << 1;

    sms_PolarToRect(sizeMag, pFftBuffer, pMag, pPhase);
    sms_ifft(sizeFft, pFftBuffer); 

    /* assume that the output array does not need to be cleared */
    /* before, this was multiplied by .5, why? */
    for(i = 0; i < sizeWaveform; i++)
        //pWaveform[i] +=  pFftBuffer[i] * pWindow[i];
        pWaveform[i] = pFftBuffer[i];

    return sizeFft;
}

/*! \brief function for a quick inverse spectrum, windowed
 * function to perform the inverse FFT, windowing the output
 * sfloat *pFMagSpectrum   input magnitude spectrum
 * sfloat *pFPhaseSpectrum input phase spectrum
 * int sizeFft             size of FFT
 * sfloat *pFWaveform      output waveform
 * int sizeWave            size of output waveform
 * sfloat *pFWindow        synthesis window
 */
int sms_invQuickSpectrumW(sfloat *pFMagSpectrum, sfloat *pFPhaseSpectrum, 
                          int sizeFft, sfloat *pFWaveform, int sizeWave,
                          sfloat *pFWindow, sfloat* pFftBuffer)
{
    int i, it2;
    int sizeMag = sizeFft >> 1;

    /* convert from polar coordinates to rectangular */
    for(i = 0; i < sizeMag; i++)
    {
        it2 = i << 1;
        pFftBuffer[it2] = pFMagSpectrum[i] * cos(pFPhaseSpectrum[i]);
        pFftBuffer[it2+1] = pFMagSpectrum[i] * sin(pFPhaseSpectrum[i]);
    }    

    /* compute IFFT */
    sms_ifft(sizeFft, pFftBuffer); 

    /* assume that the output array does not need to be cleared */
    for(i = 0; i < sizeWave; i++)
        pFWaveform[i] +=  (pFftBuffer[i] * pFWindow[i] * .5);

    return sizeMag;
}

/*! \brief convert spectrum from Rectangular to Polar form
 *              
 * \param sizeMag size of spectrum (pMag and pPhase arrays)
 * \param pRect   pointer output spectrum in rectangular form (2x sizeSpec)
 * \param pMag    pointer to sfloat array of magnitude spectrum
 * \param pPhase  pointer to sfloat array of phase spectrum
 */ 
void sms_RectToPolar(int sizeMag, sfloat *pRect, sfloat *pMag, sfloat *pPhase)
{
    int i, it2;
    sfloat fReal, fImag;

    for(i=0; i<sizeMag; i++)
    {
        it2 = i << 1;
        fReal = pRect[it2];
        fImag = pRect[it2+1];

        pMag[i] = sqrtf(fReal * fReal + fImag * fImag);
        if(pPhase)
            pPhase[i] = atan2f(fImag, fReal);
    }
}

/*! \brief convert spectrum from Rectangular to Polar form
 *              
 * \param sizeSpec size of spectrum (pMag and pPhase arrays)
 * \param pRect    pointer output spectrum in rectangular form (2x sizeSpec)
 * \param pMag     pointer to sfloat array of magnitude spectrum
 * \param pPhase   pointer to sfloat array of phase spectrum
 */ 
void sms_PolarToRect(int sizeSpec, sfloat *pRect, sfloat *pMag, sfloat *pPhase)
{
    int i, it2;
    sfloat fMag;

    for(i = 0; i<sizeSpec; i++)
    {
        it2 = i << 1;
        fMag = pMag[i];
        pRect[it2] =  fMag * cos (pPhase[i]);
        pRect[it2+1] = fMag * sin (pPhase[i]);
    }    
}

/*! \brief compute magnitude spectrum of a DFT in rectangular coordinates
 *              
 * \param sizeMag size of output Magnitude (half of input real FFT)
 * \param pInRect pointer to input DFT array (real/imag sfloats)
 * \param pOutMag pointer to of magnitude spectrum array
 */
void sms_spectrumRMS(int sizeMag, sfloat *pInRect, sfloat *pOutMag)
{
    int i, it2;
    sfloat fReal, fImag;

    for(i=0; i<sizeMag; i++)
    {
        it2 = i << 1;
        fReal = pInRect[it2];
        fImag = pInRect[it2+1];
        pOutMag[i] = sqrtf(fReal * fReal + fImag * fImag);
    }
}

