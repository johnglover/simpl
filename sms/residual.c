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
/*! \file residual.c
 * \brief main sms_residual() function
 */

#include "sms.h"

/*! \brief get the residual waveform
 *
 * \param sizeWindow       size of buffers
 * \param pSynthesis       pointer to deterministic component
 * \param pOriginal          pointer to original waveform
 * \param pResidual        pointer to output residual waveform
 * \param pWindow    pointer to windowing array
 * \return residual percentage (0 if residual was not large enough)
  \todo why is residual energy percentage computed this way? should be optional and in a seperate function
 */
int sms_residual (int sizeWindow, sfloat *pSynthesis, sfloat *pOriginal, sfloat *pResidual)
{
	static sfloat fResidualMag = 0.;
    static sfloat fOriginalMag = 0.;
	sfloat fScale = 1.;
	sfloat fCurrentResidualMag = 0.;
	sfloat fCurrentOriginalMag = 0.;
	int i;
   
	/* get residual */
	for (i=0; i<sizeWindow; i++)
	{
		pResidual[i] = pOriginal[i] - pSynthesis[i];
	}

	/* get energy of residual */
	for (i=0; i<sizeWindow; i++)
	{
		fCurrentResidualMag += (pResidual[i] * pResidual[i]);
	}

	/* if residual is big enough compute coefficients */
	if (fCurrentResidualMag) //always compute
	{
		/* get energy of original */
		for (i=0; i<sizeWindow; i++)
		{
			fCurrentOriginalMag += (pOriginal[i] * pOriginal[i]);
		}
		fOriginalMag = .5 * (fCurrentOriginalMag/sizeWindow + fOriginalMag);
		fResidualMag = .5 * (fCurrentResidualMag/sizeWindow + fResidualMag);

		/* scale residual if need to be */
		if (fResidualMag > fOriginalMag)
		{
			fScale = fOriginalMag / fResidualMag;
			for (i=0; i<sizeWindow; i++)
			{
				pResidual[i] *= fScale;
			}
		}

		return fCurrentResidualMag / fCurrentOriginalMag;
	}
	return 0;
}

/*! \brief get the residual waveform
 *
 * \param sizeWindow       size of buffers
 * \param pSynthesis       pointer to deterministic component
 * \param pOriginal          pointer to original waveform
 * \param pResidual        pointer to output residual waveform
 * \param pWindow    pointer to windowing array
 * \return  residual percentage
  \todo why is residual energy percentage computed this way? should be optional and in a seperate function
 */
int sms_residualOLD ( int sizeWindow, sfloat *pSynthesis, sfloat *pOriginal, sfloat *pResidual, sfloat *pWindow)
{
	static sfloat fResidualMag = 0, fOriginalMag = 0, *pFWindow = NULL;
	sfloat fScale = 1, fCurrentResidualMag = 0, fCurrentOriginalMag = 0;
	int i;
   
	/* get residual */
	for (i=0; i<sizeWindow; i++)
                pResidual[i] = pOriginal[i] - pSynthesis[i];

	/* get energy of residual */
	for (i=0; i<sizeWindow; i++)
		fCurrentResidualMag += fabsf( pResidual[i] * pWindow[i]);

	/* if residual is big enough compute coefficients */
//	if (fCurrentResidualMag/sizeWindow > .01)
	if (fCurrentResidualMag) //always compute
	{  
/*                 printf(" fCurrentResidualMag: %f, sizeWindow: %d, ratio: %f\n", */
/*                        fCurrentResidualMag, sizeWindow, fCurrentResidualMag/sizeWindow ); */
                
		/* get energy of original */
		for (i=0; i<sizeWindow; i++)
			fCurrentOriginalMag += fabs( pOriginal[i] * pWindow[i]);
                
		fOriginalMag = 
			.5 * (fCurrentOriginalMag/sizeWindow + fOriginalMag);
		fResidualMag = 
			.5 * (fCurrentResidualMag/sizeWindow + fResidualMag);
  
		/* scale residual if need to be */
		if (fResidualMag > fOriginalMag)
		{
			fScale = fOriginalMag / fResidualMag;
			for (i=0; i<sizeWindow; i++)
				pResidual[i] *= fScale;
		}
                
                printf("risidual mag: %f, original mag: %f \n", fCurrentResidualMag , fCurrentOriginalMag);
                return (fCurrentResidualMag / fCurrentOriginalMag);
	}
                else printf("whaaaat not big enough: fCurrentResidualMag: %f, sizeWindow: %d, ratio: %f\n",
                            fCurrentResidualMag, sizeWindow, fCurrentResidualMag/sizeWindow );
        
	return (0);
}

