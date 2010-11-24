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
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, M  02111-1307  USA
 * 
 */
/*! \file stocAnalysis.c
 * \brief stochastic analysis using spectral analysis and approximation
 */
#include "sms.h"

#define ENV_THRESHOLD     .01 /* \todo was this value for type shorts?? */

/*! \brief main function for the stochastic analysis
 * \param sizeWindow   size of buffer
 * \param pResidual    pointer to residual signal
 * \param pWindow      pointer to windowing array
 * \param pSmsData     pointer to output SMS data
 * \param pAnalParams  point to analysis parameters
 * \return 0 on success, -1 on error
 */
int sms_stocAnalysis(int sizeWindow, sfloat *pResidual, sfloat *pWindow, 
                     SMS_Data *pSmsData, SMS_AnalParams* pAnalParams)
{
    int i;
    sfloat fMag = 0.0;
    sfloat fStocNorm;

    sms_spectrumMag(sizeWindow, pResidual, pWindow, pAnalParams->sizeStocMagSpectrum,
                    pAnalParams->stocMagSpectrum, pAnalParams->fftBuffer);

    sms_spectralApprox(pAnalParams->stocMagSpectrum, pAnalParams->sizeStocMagSpectrum, 
                       pAnalParams->sizeStocMagSpectrum, pSmsData->pFStocCoeff, 
                       pSmsData->nCoeff, pSmsData->nCoeff,
                       pAnalParams->approxEnvelope);

    /* get energy of spectrum  */
    for(i = 0; i < pAnalParams->sizeStocMagSpectrum; i++)
        fMag += (pAnalParams->stocMagSpectrum[i] * pAnalParams->stocMagSpectrum[i]);

    *pSmsData->pFStocGain = fMag / pAnalParams->sizeStocMagSpectrum;
    fStocNorm = 1. / *pSmsData->pFStocGain;

    return 0;
}

