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
/*! \file synthesis.c
 * \brief main synthesis routines
 */
#include "sms.h"

/*! \brief synthesis of one frame of the deterministic component using the IFFT
 *
 * \param pSmsData pointer to SMS data structure frame
 * \param pSynthParams pointer to structure of synthesis parameters
 */
static void SineSynthIFFT (SMS_Data *pSmsData, SMS_SynthParams *pSynthParams)
{
        int sizeFft = pSynthParams->sizeHop << 1; 
        int iHalfSamplingRate = pSynthParams->iSamplingRate >> 1;
        int sizeMag = pSynthParams->sizeHop;
        int nBins = 8;
        int nTracks = pSmsData->nTracks;
        int iFirstBin, k, i, l, b;
        sfloat fMag=0.0, fFreq=0.0, fPhase=0.0, fLoc, fSin, fCos, fBinRemainder, 
                fTmp, fNewMag,  fIndex;
        sfloat fSamplingPeriod = 1.0 / pSynthParams->iSamplingRate;
        memset (pSynthParams->pSpectra, 0, sizeFft * sizeof(sfloat));
        for (i = 0; i < nTracks; i++)
        {
                if (((fMag = pSmsData->pFSinAmp[i]) > 0) &&
                    ((fFreq = (pSmsData->pFSinFreq[i])) < iHalfSamplingRate))
                {
                        /* \todo maybe this check can be removed if the SynthParams->prevFrame gets random
                           phases in sms_initSynth? */
                        if (pSynthParams->prevFrame.pFSinAmp[i] <= 0)
                                pSynthParams->prevFrame.pFSinPha[i] = TWO_PI * sms_random();

                        // fMag = sms_dBToMag (fMag);
                        fTmp = pSynthParams->prevFrame.pFSinPha[i] +
                                TWO_PI * fFreq * fSamplingPeriod * sizeMag;
                        fPhase = fTmp - floor(fTmp * INV_TWO_PI) * TWO_PI;
                        fLoc = sizeFft * fFreq  * fSamplingPeriod;
                        iFirstBin = (int) fLoc - 3;
                        fBinRemainder = fLoc - floor (fLoc);
                        fSin = sms_sine (fPhase);
                        fCos = sms_sine (fPhase + PI_2);
                        for (k = 1, l = iFirstBin; k <= nBins; k++, l++)
                        {
                                fIndex = (k - fBinRemainder);
                                if (fIndex > 7.999) fIndex = 0;
                                fNewMag = fMag * sms_sinc (fIndex);
                                if (l > 0 && l < sizeMag)
                                {
                                        pSynthParams->pSpectra[l*2+1] += fNewMag * fCos;
                                        pSynthParams->pSpectra[l*2] += fNewMag * fSin;
                                }
                                else if (l == 0)
                                {
                                        pSynthParams->pSpectra[0] += 2 * fNewMag * fSin;
                                }
                                else if (l < 0)
                                {
                                        b = abs(l);
                                        pSynthParams->pSpectra[b*2+1] -= fNewMag * fCos;
                                        pSynthParams->pSpectra[b*2] += fNewMag * fSin;
                                }
                                else if (l > sizeMag)
                                {
                                        b = sizeMag - (l - sizeMag);
                                        pSynthParams->pSpectra[b*2+1] -= fNewMag * fCos;
                                        pSynthParams->pSpectra[b*2] += fNewMag * fSin;
                                }
                                else if (l == sizeMag)
                                {
                                        pSynthParams->pSpectra[1] += 2 * fNewMag * fSin;
                                }
                        }
                }
                pSynthParams->prevFrame.pFSinAmp[i] = fMag;
                pSynthParams->prevFrame.pFSinPha[i] = fPhase;
                pSynthParams->prevFrame.pFSinFreq[i] = fFreq;
        }

        sms_ifft(sizeFft, pSynthParams->pSpectra);

        for(i = 0, k = sizeMag; i < sizeMag; i++, k++) 
                pSynthParams->pSynthBuff[i] += pSynthParams->pSpectra[k] * pSynthParams->pFDetWindow[i];
        for(i= sizeMag, k = 0; i < sizeFft; i++, k++) 
                pSynthParams->pSynthBuff[i] +=  pSynthParams->pSpectra[k] * pSynthParams->pFDetWindow[i];

}

/*! \brief synthesis of one frame of the deterministic component using the IFFT
 *
 * 
 * I made this function trying to pull the ifft out of the SineSynthIFFT
 * that part of the code is in the main sms_synthesize function below
 * Next would be sms_stochastic, so they could be summed and synthesized
 * with only one fft, but the problem is that it needs to be 'pre-inverse-windowed'
 * before the stochastic can be summed.
 *
 * \param pSmsData pointer to SMS data structure frame
 * \param pSynthParams pointer to structure of synthesis parameters
 */
void sms_deterministic (SMS_Data *pSmsData, SMS_SynthParams *pSynthParams)
{
        int sizeFft = pSynthParams->sizeHop << 1; 
        int iHalfSamplingRate = pSynthParams->iSamplingRate >> 1;
        int sizeSpec = pSynthParams->sizeHop;
        int nBins = 8;
        int nTracks = pSmsData->nTracks;
        int iFirstBin, k, i, l, b;
        sfloat fMag=0.0, fFreq=0.0, fPhase=0.0, fLoc, fSin, fCos, fBinRemainder, 
                fTmp, fNewMag,  fIndex;
        sfloat fSamplingPeriod = 1.0 / pSynthParams->iSamplingRate;
        printf("| in deterministic | ");

        for (i = 0; i < nTracks; i++)
        {
                if (((fMag = pSmsData->pFSinAmp[i]) > 0) &&
                    (fFreq = pSmsData->pFSinFreq[i] < iHalfSamplingRate)) /* \todo why does this allow f < 0 */
                {
                        /* \todo maybe this check can be removed if the SynthParams->prevFrame gets random
                           phases in sms_initSynth? */
                        if (pSynthParams->prevFrame.pFSinAmp[i] <= 0)
                                pSynthParams->prevFrame.pFSinPha[i] = TWO_PI * sms_random();

                        //fMag = sms_dBToMag (fMag);
                        fTmp = pSynthParams->prevFrame.pFSinPha[i] +
                                TWO_PI * fFreq * fSamplingPeriod * sizeSpec;
                        fPhase = fTmp - floor(fTmp * INV_TWO_PI) * TWO_PI;
                        fLoc = sizeFft * fFreq  * fSamplingPeriod;
                        iFirstBin = (int) fLoc - 3;
                        fBinRemainder = fLoc - floor (fLoc);
                        fSin = sms_sine (fPhase);
                        fCos = sms_sine (fPhase + PI_2);
                        for (k = 1, l = iFirstBin; k <= nBins; k++, l++)
                        {
                                fIndex = (k - fBinRemainder);
                                if (fIndex > 7.999) fIndex = 0;
                                fNewMag = fMag * sms_sinc (fIndex);
                                if (l > 0 && l < sizeSpec)
                                {
                                        pSynthParams->pSpectra[l*2+1] += fNewMag * fCos;
                                        pSynthParams->pSpectra[l*2] += fNewMag * fSin;
                                }
                                else if (l == 0)
                                {
                                        pSynthParams->pSpectra[0] += 2 * fNewMag * fSin;
                                }
                                else if (l < 0)
                                {
                                        b = abs(l);
                                        pSynthParams->pSpectra[b*2+1] -= fNewMag * fCos;
                                        pSynthParams->pSpectra[b*2] += fNewMag * fSin;
                                }
                                else if (l > sizeSpec)
                                {
                                        b = sizeSpec - (l - sizeSpec);
                                        pSynthParams->pSpectra[b*2+1] -= fNewMag * fCos;
                                        pSynthParams->pSpectra[b*2] += fNewMag * fSin;
                                }
                                else if (l == sizeSpec)
                                {
                                        pSynthParams->pSpectra[1] += 2 * fNewMag * fSin;
                                }
                        }
                }
                pSynthParams->prevFrame.pFSinAmp[i] = fMag;
                pSynthParams->prevFrame.pFSinPha[i] = fPhase;
                pSynthParams->prevFrame.pFSinFreq[i] = fFreq;
        }

/*         sms_ifft(sizeFft, pSynthParams->pSpectra); */

/*         for(i = 0, k = sizeMag; i < sizeMag; i++, k++)  */
/*                 pSynthParams->pSynthBuff[i] += pSynthParams->pSpectra[k] * pSynthParams->pFDetWindow[i]; */
/*         for(i= sizeMag, k = 0; i < sizeFft; i++, k++)  */
/*                 pSynthParams->pSynthBuff[i] +=  pSynthParams->pSpectra[k] * pSynthParams->pFDetWindow[i]; */

}

/*! \brief synthesis of one frame of the stochastic component by apprimating phases
 * 
 * computes a linearly interpolated spectral envelope to fit the correct number of output
 * audio samples. Phases are generated randomly.  
 *
 * \param pSmsData pointer to the current SMS frame
 * \param pSynthParams pointer to a strucure of synthesis parameters
 * \return 
 * \todo cleanup returns and various constant multipliers. check that approximation is ok
 */
static int StocSynthApprox (SMS_Data *pSmsData, SMS_SynthParams *pSynthParams)
{       
        int i, sizeSpec1Used;
        int sizeSpec1 = pSmsData->nCoeff;
        int sizeSpec2 = pSynthParams->sizeHop;
        int sizeFft = pSynthParams->sizeHop << 1; /* 50% overlap, so sizeFft is 2x sizeHop */
        float fStocGain;
        /* if no gain or no coefficients return  */
        if (*(pSmsData->pFStocGain) <= 0)
                return 0;

        // *(pSmsData->pFStocGain) = sms_dBToMag(*(pSmsData->pFStocGain));

        /* \todo check why this was here */
        /* for (i = 0; i < sizeSpec1; i++)
                 pSmsData->pFStocCoeff[i] *= 2 * *(pSmsData->pFStocGain) ;
        */
        sizeSpec1Used = sizeSpec1 * pSynthParams->iSamplingRate /
                pSynthParams->iOriginalSRate;
        /* sizeSpec1Used cannot be more than what is available  \todo check by graph */
        if(sizeSpec1Used  > sizeSpec1) sizeSpec1Used = sizeSpec1;
        //printf("iSamplingRate: %d, iOriginalSRate: %d, sizeSpec1: %d, sizeSpec1Used: %d, sizeSpec2: %d \n",
        //         pSynthParams->iSamplingRate, pSynthParams->iOriginalSRate, sizeSpec1, sizeSpec1Used, sizeSpec2);
        sms_spectralApprox (pSmsData->pFStocCoeff, sizeSpec1, sizeSpec1Used,
                        pSynthParams->pMagBuff, sizeSpec2, sizeSpec1Used);

        /* generate random phases */
        for (i = 0; i < sizeSpec2; i++)
                pSynthParams->pPhaseBuff[i] =  TWO_PI * sms_random();

        sms_invQuickSpectrumW (pSynthParams->pMagBuff, pSynthParams->pPhaseBuff,
                               sizeFft, pSynthParams->pSynthBuff, sizeFft,
                               pSynthParams->pFStocWindow);
        return 1;
}

/*! \brief synthesis of one frame of the residual component, modeled as stochastic,
 * by approximating phases
 * 
 * computes a linearly interpolated spectral envelope to fit the correct number of output
 * audio samples. Phases are generated randomly.  
 * 
 * \todo needs to be pre-windowed to match the windowing effect of the deterministic,
 * so when they are summed then they can be ifft'ed and overlap-added properly
 *
 * \param pSmsData pointer to the current SMS frame
 * \param pSynthParams pointer to a strucure of synthesis parameters
 * \return 
 */
int sms_stochastic (SMS_Data *pSmsData, SMS_SynthParams *pSynthParams)
{
        int i, sizeOriginalSpecUsed;
        int sizeOriginalSpec = pSmsData->nCoeff;
        int sizeSpec = pSynthParams->sizeHop;
        int sizeFft = pSynthParams->sizeHop << 1; /* 50% overlap, so sizeFft is 2x sizeHop */

        /* if no gain or no coefficients return  */
        if (*(pSmsData->pFStocGain) <= 0)
                return (0);

/*         *(pSmsData->pFStocGain) = sms_dBToMag(*(pSmsData->pFStocGain)); */

        /* scale the coefficients to normal amplitude */
        /*! \todo why is it also multiplied by 2? Why aren't the coeffecients just stored with gain already multiplied?*/
/*         for (i = 0; i < sizeOriginalSpec; i++) */
/*                 pSmsData->pFStocCoeff[i] *= 2 * *(pSmsData->pFStocGain) ; */

        sizeOriginalSpecUsed = sizeOriginalSpec * pSynthParams->iSamplingRate / 
                pSynthParams->iOriginalSRate;
        /* sizeOriginalSpecUsed cannot be more than what is available  */
        if(sizeOriginalSpecUsed  > sizeOriginalSpec) sizeOriginalSpecUsed = sizeOriginalSpec;
        sms_spectralApprox (pSmsData->pFStocCoeff, sizeOriginalSpec, sizeOriginalSpecUsed,
                        pSynthParams->pMagBuff, sizeSpec, sizeOriginalSpecUsed);

        /* generate random phases */
        for (i = 0; i < sizeSpec; i++)
                pSynthParams->pPhaseBuff[i] =  TWO_PI * sms_random();

        /* \todo first  multiply the pMagBuff by a window in order to properly un-window below */
        sms_PolarToRect(sizeSpec, pSynthParams->pSpectra, pSynthParams->pMagBuff, pSynthParams->pPhaseBuff);
        return (0);
}

/*! \brief  synthesizes one frame of SMS data
 *
 * \param pSmsData      input SMS data
 * \param pFSynthesis      output sound buffer  
 * \param pSynthParams   synthesis parameters
 */
void sms_synthesize(SMS_Data *pSmsData, sfloat *pFSynthesis,
                    SMS_SynthParams *pSynthParams)
{
        int i, k;
        int sizeHop = pSynthParams->sizeHop;
        int sizeFft = sizeHop << 1;
  
        memcpy (pSynthParams->pSynthBuff, (sfloat *)(pSynthParams->pSynthBuff+sizeHop),
                sizeof(sfloat) * sizeHop);
        memset (pSynthParams->pSynthBuff+sizeHop, 0, sizeof(sfloat) * sizeHop);
        
        /* convert mags from linear to db */
        sms_arrayMagToDB(pSmsData->nTracks, pSmsData->pFSinAmp);

        /* decide which combo of synthesis methods to use */
        if(pSynthParams->iSynthesisType == SMS_STYPE_ALL)
        {
                if(pSynthParams->iDetSynthType == SMS_DET_IFFT &&
                   pSynthParams->iStochasticType == SMS_STOC_IFFT)
                {
                        memset (pSynthParams->pSpectra, 0, sizeFft  * sizeof(sfloat));
                        sms_deterministic(pSmsData, pSynthParams);
                        sms_ifft(sizeFft, pSynthParams->pSpectra);

                        for(i = 0, k = sizeHop; i < sizeHop; i++, k++)
                                pSynthParams->pSynthBuff[i] += pSynthParams->pSpectra[k] * pSynthParams->pFDetWindow[i];
                        for(i= sizeHop, k = 0; i < sizeFft; i++, k++)
                                pSynthParams->pSynthBuff[i] +=  pSynthParams->pSpectra[k] * pSynthParams->pFDetWindow[i];

                        //sms_invSpectrum(sizeFft, pSynthParams->pSynthBuff, 


                }
                else /* can't use combo  IFFT, synthesize seperately and sum */
                {
                        if(pSynthParams->iDetSynthType == SMS_DET_IFFT)
                                SineSynthIFFT (pSmsData, pSynthParams);
                        else /*pSynthParams->iDetSynthType == SMS_DET_SIN*/
                        {
                                sms_sineSynthFrame (pSmsData, pSynthParams->pSynthBuff+sizeHop, pSynthParams->sizeHop,
                                                    &(pSynthParams->prevFrame), pSynthParams->iSamplingRate);
                        }
                        StocSynthApprox (pSmsData, pSynthParams);
                }

        }
        else if(pSynthParams->iSynthesisType == SMS_STYPE_DET)
        {
                if(pSynthParams->iDetSynthType == SMS_DET_IFFT)
                        SineSynthIFFT (pSmsData, pSynthParams);
                else /*pSynthParams->iDetSynthType == SMS_DET_SIN*/
                {
                        sms_sineSynthFrame (pSmsData, pSynthParams->pSynthBuff+sizeHop, pSynthParams->sizeHop,
                                        &(pSynthParams->prevFrame), pSynthParams->iSamplingRate);
                }
        }
        else /* pSynthParams->iSynthesisType == SMS_STYPE_STOC */
                StocSynthApprox(pSmsData, pSynthParams);

        /* de-emphasize the sound */
        for(i = 0; i < sizeHop; i++)
			pFSynthesis[i] = sms_deEmphasis(pSynthParams->pSynthBuff[i+sizeHop], pSynthParams);
}

