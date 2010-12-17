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
 * \param sizeWindow size of buffers
 * \param pSynthesis pointer to deterministic component
 * \param pOriginal  pointer to original waveform
 * \param pResidual  pointer to output residual waveform
 * \param pWindow    pointer to windowing array
 * \return residual percentage (0 if residual was not large enough)
 */
int sms_residual(int sizeWindow, sfloat *pSynthesis, sfloat *pOriginal, sfloat *pResidual, sfloat *pWindow)
{
    static sfloat fResidualMag = 0.;
    static sfloat fOriginalMag = 0.;
    sfloat fScale = 1.;
    sfloat fCurrentResidualMag = 0.;
    sfloat fCurrentOriginalMag = 0.;
    int i;

    /* get residual */
    for (i=0; i<sizeWindow; i++)
        pResidual[i] = pOriginal[i] - pSynthesis[i];

    /* get energy of residual */
    for (i=0; i<sizeWindow; i++)
        fCurrentResidualMag += (pResidual[i] * pResidual[i]);

    /* if residual is big enough compute coefficients */
    if (fCurrentResidualMag)
    {  
        /* get energy of original */
        for (i=0; i<sizeWindow; i++)
            fCurrentOriginalMag += (pOriginal[i] * pOriginal[i]);

        fOriginalMag = .5 * (fCurrentOriginalMag/sizeWindow + fOriginalMag);
        fResidualMag = .5 * (fCurrentResidualMag/sizeWindow + fResidualMag);

        /* scale residual if need to be */
        if (fResidualMag > fOriginalMag)
        {
            fScale = fOriginalMag / fResidualMag;
            for (i=0; i<sizeWindow; i++)
                pResidual[i] *= fScale;
        }

        return fCurrentResidualMag / fCurrentOriginalMag;
    }
    return 0;
}

