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
 * \param sizeWindow         size of buffer
 * \param pResidual      pointer to residual signal
 * \param pWindow      pointer to windowing array
 * \param pSmsData        pointer to output SMS data
 * \return 0 on success, -1 on error
 */
int sms_stocAnalysis ( int sizeWindow, sfloat *pResidual, sfloat *pWindow, SMS_Data *pSmsData)
{
        int i;
        sfloat fMag = 0.0;
        sfloat fStocNorm;

        static sfloat *pMagSpectrum;
        static int sizeWindowStatic = 0;
        static int sizeFft = 0;
        static int sizeMag = 0;
        /* update array sizes if sizeWindow is new */
        if (sizeWindowStatic != sizeWindow)
        {
                if(sizeWindowStatic != 0) free(pMagSpectrum);
                sizeWindowStatic = sizeWindow;
                sizeFft = sms_power2(sizeWindow);
                sizeMag = sizeFft >> 1;
                if((pMagSpectrum = (sfloat *) calloc(sizeMag, sizeof(sfloat))) == NULL)
                {
                        sms_error("sms_stocAnalysis: error allocating memory for pMagSpectrum");
                        return -1;
                }
        }
        
        sms_spectrumMag (sizeWindow, pResidual, pWindow, sizeMag, pMagSpectrum);
 
        sms_spectralApprox (pMagSpectrum, sizeMag, sizeMag, pSmsData->pFStocCoeff, 
                            pSmsData->nCoeff, pSmsData->nCoeff);
  
        /* get energy of spectrum  */
        for (i = 0; i < sizeMag; i++)
                fMag += (pMagSpectrum[i] * pMagSpectrum[i]);
        *pSmsData->pFStocGain = fMag / sizeMag;
        fStocNorm = 1. / *pSmsData->pFStocGain;

        /* normalize envelope */
        /* \todo what good is this scaling, it is only being undone in resynthesis */
/*         for (i = 0; i <  pSmsData->nCoeff; i++) */
/*                 pSmsData->pFStocCoeff[i] *= fStocNorm; */
    
        // *pSmsData->pFStocGain = sms_magToDB(*pSmsData->pFStocGain);
	return(0);
}

