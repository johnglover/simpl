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
/*! \file transforms.c 
 * \brief routines for different Fast Fourier Transform Algorithms
 *
 */

#include "sms.h"
#include "OOURA.h"

static int ip[NMAXSQRT +2];
static sfloat w[NMAX * 5 / 4];

/*! \brief Forward Fast Fourier Transform
 *
 * function to call the OOURA routines to calculate
 * the forward FFT. Operation is in place.
 * \todo if sizeFft != power of 2, there is a silent crash.. cuidado!
 *
 * \param sizeFft size of the FFT in samples (must be a power of 2 >= 2)
 * \param pArray  pointer to real array (n >= 2, n = power of 2)
 */
void sms_fft(int sizeFft, sfloat *pArray)
{ 
    rdft(sizeFft, 1, pArray, ip, w);
}

/*! \brief Inverse Forward Fast Fourier Transform
 *
 * function to call the OOURA routines to calculate
 * the Inverse FFT. Operation is in place.
 *
 * \param sizeFft size of the FFT in samples (must be a power of 2 >= 2)
 * \param pArray  pointer to real array (n >= 2, n = power of 2)
 */
void sms_ifft(int sizeFft, sfloat *pArray)
{ 
    rdft(sizeFft, -1, pArray, ip, w);
}
