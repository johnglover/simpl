/* 
 * Copyright (c) 2008 MUSIC TECHNOLOGY GROUP (MTG)
 *                    UNIVERSITAT POMPEU FABRA 
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

/*! \file windows.c
 * \brief functions for creating various windows
 * 
 * Use sms_getWindow() for selecting which window will be made
 */
#include "sms.h"

/* \brief scale a window by its integral (numeric quadrature)
 *
 * In order to get a normalized magnitude spectrum (ex. Fourier analysis
 * of a sinusoid with linear magnitude 1 gives one peak of magnitude 1 in
 * the frequency domain), the spectrum windowing function should be 
 * normalized by its area under the curve.  
 *
 * \param sizeWindow the size of the window
 * \param pWindow pointer to an array that will hold the window
 */
void sms_scaleWindow(int sizeWindow, sfloat *pWindow)
{
    int i;
    sfloat fSum = 0;
    sfloat fScale;

    for(i = 0; i < sizeWindow; i++) 
        fSum += pWindow[i];

    fScale =  2. / fSum;

    for(i = 0; i < sizeWindow; i++)
        pWindow[i] *= fScale;
}

/*! \brief window to be used in the IFFT synthesis
 * 
 * contains both an inverse Blackman-Harris and triangular window.
 *
 * \todo read X. Rodet, Ph. Depalle, "Spectral Envelopes and Inverse FFT
 * Synthesis." Proc. 93rd AES Convention, October 1992
 * \param sizeWindow the size of the window
 * \param pFWindow pointer to an array that will hold the window
 */
void IFFTwindow(int sizeWindow, sfloat *pFWindow)
{
    int i;
    sfloat a0 = .35875, a1 = .48829, a2 = .14128, a3 = .01168;
    double fConst = TWO_PI / sizeWindow, fIncr = 2.0 /sizeWindow, fVal = 0;

    /* compute inverse of Blackman-Harris 92dB window */
    for(i = 0; i < sizeWindow; i++) 
    {
        pFWindow[i] = 1 / (a0 - a1 * cos(fConst * i) +
                      a2 * cos(fConst * 2 * i) - a3 * cos(fConst * 3 * i));
    }

    /* scale function by a triangular */
    for(i = 0; i < sizeWindow / 2; i++)
    {
        pFWindow[i] = fVal * pFWindow[i]  / 2.787457;
        fVal += fIncr;
    }
    for(i = sizeWindow / 2; i < sizeWindow; i++)
    {
        pFWindow[i] = fVal * pFWindow[i]  / 2.787457;
        fVal -= fIncr;
    }
}

/*! \brief BlackmanHarris window with 62dB rolloff
 * 
 * \todo where did these come from?
 * \param sizeWindow the size of the window
 * \param pFWindow pointer to an array that will hold the window
 */
void BlackmanHarris62(int sizeWindow, sfloat *pFWindow)
{
    int i;
    double fSum = 0;
    /* for 3 term -62.05 */
    sfloat a0 = .44959, a1 = .49364, a2 = .05677; 
    double fConst = TWO_PI / sizeWindow;

    /* compute window */
    for(i = 0; i < sizeWindow; i++) 
    {
        fSum += pFWindow[i] = a0 - a1 * cos(fConst * i) +
                a2 * cos(fConst * 2 * i);
    }
}

/*! \brief BlackmanHarris window with 70dB rolloff
 * 
 * \param sizeWindow the size of the window
 * \param pFWindow pointer to an array that will hold the window
 */
void BlackmanHarris70(int sizeWindow, sfloat *pFWindow)
{
    int i;
    double fSum = 0;
    /* for 3 term -70.83 */
    sfloat a0 = .42323, a1 = .49755, a2 = .07922;
    double fConst = TWO_PI / sizeWindow;

    /* compute window */
    for(i = 0; i < sizeWindow; i++) 
    {
        fSum += pFWindow[i] = a0 - a1 * cos(fConst * i) +
                a2 * cos(fConst * 2 * i);
    }
}

/*! \brief BlackmanHarris window with 74dB rolloff
 * 
 * \param sizeWindow the size of the window
 * \param pFWindow pointer to an array that will hold the window
 */
void BlackmanHarris74(int sizeWindow, sfloat *pFWindow)
{
    int i;
    double fSum = 0;
    /* for -74dB  from the Nuttall paper */
    sfloat a0 = .40217, a1 = .49703, a2 = .09892, a3 = .00188;
    double fConst = TWO_PI / sizeWindow;

    /* compute window */
    for(i = 0; i < sizeWindow; i++) 
    {
        fSum += pFWindow[i] = a0 - a1 * cos(fConst * i) +
                a2 * cos(fConst * 2 * i) + a3 * cos(fConst * 3 * i);
    }
}

/*! \brief BlackmanHarris window with 92dB rolloff
 * 
 * \param sizeWindow the size of the window
 * \param pFWindow pointer to an array that will hold the window
 */
void BlackmanHarris92(int sizeWindow, sfloat *pFWindow)
{
    int i;
    double fSum = 0;
    /* for -92dB */
    sfloat a0 = .35875, a1 = .48829, a2 = .14128, a3 = .01168;
    double fConst = TWO_PI / sizeWindow;

    /* compute window */
    for(i = 0; i < sizeWindow; i++) 
    {
        fSum += pFWindow[i] = a0 - a1 * cos(fConst * i) +
                a2 * cos(fConst * 2 * i) + a3 * cos(fConst * 3 * i);
    }
}

/*! \brief default BlackmanHarris window (70dB rolloff)
 * 
 * \param sizeWindow the size of the window
 * \param pFWindow pointer to an array that will hold the window
 */
void BlackmanHarris(int sizeWindow, sfloat *pFWindow)
{
    BlackmanHarris70(sizeWindow, pFWindow);
}

/*! \brief Hamming window
 *
 * \param sizeWindow window size
 * \param pWindow    window array
 */
void Hamming(int sizeWindow, sfloat *pWindow)
{
    int i;
    sfloat fSum = 0;

    for(i = 0; i < sizeWindow; i++) 
    {
        fSum += pWindow[i] = 0.53836 - 0.46164*cos(TWO_PI*i/(sizeWindow-1));
    }
}

/*! \brief Hanning window
 *
 * \param sizeWindow window size
 * \param pWindow    window array
 */
void Hanning(int sizeWindow, sfloat *pWindow)
{
    int i;
    for(i = 0; i < sizeWindow; i++) 
        pWindow[i] = (sin(PI*i/(sizeWindow-1)))*(sin(PI*i/(sizeWindow-1)));
}

/*! \brief main function for getting various windows
 *
 * \todo note on window scales
 * 
 * \see SMS_WINDOWS for the different window types available
 * \param sizeWindow  window size
 * \param pFWindow    window array
 * \param iWindowType the desired window type defined by #SMS_WINDOWS 
 */
void sms_getWindow(int sizeWindow, sfloat *pFWindow, int iWindowType)
{
    switch(iWindowType)
    {
        case SMS_WIN_BH_62: 
            BlackmanHarris62(sizeWindow, pFWindow);
            break;
        case SMS_WIN_BH_70: 
            BlackmanHarris70(sizeWindow, pFWindow);            
            break;
        case SMS_WIN_BH_74: 
            BlackmanHarris74(sizeWindow, pFWindow);
            break;
        case SMS_WIN_BH_92: 
            BlackmanHarris92(sizeWindow, pFWindow);
            break;
        case SMS_WIN_HAMMING: 
            Hamming(sizeWindow, pFWindow);
            break;
        case SMS_WIN_HANNING: 
            Hanning(sizeWindow, pFWindow);
            break;
        case SMS_WIN_IFFT: 
            IFFTwindow(sizeWindow, pFWindow);
            break;
        default:
            BlackmanHarris(sizeWindow, pFWindow);
    }
}

/*! \brief apply a window and center around sample 0
 *
 * function to center a waveform around sample 0, also known
 * as 'zero-phase windowing'.  Half the samples are at the beginning,
 * half at the end, with the remaining samples  (sizeFft-sizeWindow) 
 * in the middle (zero-padding for an interpolated spectrum).
 *
 * \todo do I need to garuntee that sizeWindow is odd-lengthed?
 *
 * \param sizeWindow size of waveform/waveform
 * \param pWaveform  pointer to  waveform
 * \param pWindow    pointer to window
 * \param sizeFft    size of FFT
 * \param pFftBuffer pointer to FFT buffer
 */
void sms_windowCentered(int sizeWindow, sfloat *pWaveform, sfloat *pWindow, 
                        int sizeFft, sfloat *pFftBuffer)
{
    int iMiddleWindow = (sizeWindow+1) >> 1; 
    int iOffset = sizeFft - (iMiddleWindow - 1);
    int i;

    for(i=0; i<iMiddleWindow-1; i++)
        pFftBuffer[iOffset + i] =  pWindow[i] * pWaveform[i];

    iOffset = iMiddleWindow - 1;

    for(i=0; i<iMiddleWindow; i++)
        pFftBuffer[i] = pWindow[iOffset + i] * pWaveform[iOffset + i];
}
