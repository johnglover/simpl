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
/*! \file fileIO.c
 * \brief SMS file input and output
 */

#include "sms.h"

/*! \brief file identification constant
 * 
 * constant number that is first within SMS_Header, in order to correctly
 * identify an SMS file when read.  
 */
#define SMS_MAGIC 767  

static char pChTextString[1000]; /*!< string to store analysis parameters in sms header */

/*! \brief initialize the header structure of an SMS file
 *
 * \param pSmsHeader    header for SMS file
 */
void sms_initHeader (SMS_Header *pSmsHeader)
{    
    pSmsHeader->iSmsMagic = SMS_MAGIC;
    pSmsHeader->iHeadBSize =  sizeof(SMS_Header);
    pSmsHeader->nFrames = 0;
    pSmsHeader->iFrameBSize = 0;
    pSmsHeader->iFormat = SMS_FORMAT_H;
    pSmsHeader->iFrameRate = 0;
    pSmsHeader->iStochasticType = SMS_STOC_APPROX;
    pSmsHeader->nTracks = 0;
    pSmsHeader->nStochasticCoeff = 0;
    pSmsHeader->nEnvCoeff = 0;
    pSmsHeader->iMaxFreq = 0;
    pSmsHeader->fResidualPerc = 0;
    pSmsHeader->nTextCharacters = 0;
    pSmsHeader->pChTextCharacters = NULL;    
}

/*! \brief fill an SMS header with necessary information for storage
 * 
 * copies parameters from SMS_AnalParams, along with other values
 * so an SMS file can be stored and correctly synthesized at a later
 * time. This is somewhat of a convenience function.
 *
 * sms_initAnal() should be done first to properly set everything.
 *
 * \param pSmsHeader    header for SMS file (to be stored)
 * \param pAnalParams   structure of analysis parameters
 * \param pProgramString pointer to a string containing the name of the program that made the analysis data
 */
void sms_fillHeader (SMS_Header *pSmsHeader, SMS_AnalParams *pAnalParams,
        char *pProgramString)
{
    sms_initHeader (pSmsHeader);
    pSmsHeader->nFrames = pAnalParams->nFrames;
    pSmsHeader->iFormat = pAnalParams->iFormat;
    pSmsHeader->iFrameRate = pAnalParams->iFrameRate;
    pSmsHeader->iStochasticType = pAnalParams->iStochasticType;
    pSmsHeader->nTracks = pAnalParams->nTracks;
    pSmsHeader->iSamplingRate = pAnalParams->iSamplingRate;
    if(pAnalParams->iStochasticType == SMS_STOC_NONE)
        pSmsHeader->nStochasticCoeff = 0;
    else
        pSmsHeader->nStochasticCoeff = pAnalParams->nStochasticCoeff;
    pSmsHeader->iEnvType = pAnalParams->specEnvParams.iType;
    pSmsHeader->nEnvCoeff = pAnalParams->specEnvParams.nCoeff;
    pSmsHeader->iMaxFreq = (int) pAnalParams->fHighestFreq;
    pSmsHeader->iFrameBSize = sms_frameSizeB(pSmsHeader);
    sprintf (pChTextString, 
            "created by %s with parameters: format %d, soundType %d, "
            "analysisDirection %d, windowSize %.2f,"
            " windowType %d, frameRate %d, highestFreq %.2f, minPeakMag %.2f,"
            " refHarmonic %d, minRefHarmMag %.2f, refHarmMagDiffFromMax %.2f,"
            " defaultFund %.2f, lowestFund %.2f, highestFund %.2f, nGuides %d,"
            " nTracks %d, freqDeviation %.2f, peakContToGuide %.2f,"
            " fundContToGuide %.2f, cleanTracks %d, iMinTrackLength %d,"
            "iMaxSleepingTime %d, stochasticType %d, nStocCoeff %d\n"
            "iEnvType: %d, nEnvCoeff: %d",     
            pProgramString,
            pAnalParams->iFormat, pAnalParams->iSoundType,
            pAnalParams->iAnalysisDirection, pAnalParams->fSizeWindow, 
            pAnalParams->iWindowType, pAnalParams->iFrameRate,
            pAnalParams->fHighestFreq, pAnalParams->fMinPeakMag,
            pAnalParams->iRefHarmonic, pAnalParams->fMinRefHarmMag, 
            pAnalParams->fRefHarmMagDiffFromMax,  
            pAnalParams->fDefaultFundamental, pAnalParams->fLowestFundamental,
            pAnalParams->fHighestFundamental, pAnalParams->nGuides,
            pAnalParams->nTracks, pAnalParams->fFreqDeviation, 
            pAnalParams->fPeakContToGuide, pAnalParams->fFundContToGuide,
            pAnalParams->iCleanTracks, pAnalParams->iMinTrackLength,
            pAnalParams->iMaxSleepingTime,  pAnalParams->iStochasticType,
            pAnalParams->nStochasticCoeff, pSmsHeader->iEnvType, pSmsHeader->nEnvCoeff);

    pSmsHeader->nTextCharacters = strlen (pChTextString) + 1;
    pSmsHeader->pChTextCharacters = (char *) pChTextString;
}

/*! \brief write SMS header to file
 *
 * \param pChFileName      file name for SMS file
 * \param pSmsHeader header for SMS file
 * \param ppSmsFile     (double pointer to)  file to be created
 * \return error code \see SMS_WRERR in SMS_ERRORS 
 */
int sms_writeHeader (char *pChFileName, SMS_Header *pSmsHeader, 
        FILE **ppSmsFile)
{
    int iVariableSize = 0;

    if (pSmsHeader->iSmsMagic != SMS_MAGIC)
    {
        sms_error("not an SMS file");
        return(-1);
    }
    if ((*ppSmsFile = fopen (pChFileName, "w+")) == NULL)
    {
        sms_error("cannot open file for writing");
        return(-1);
    }   
    /* check variable size of header */
    /*  iVariableSize = sizeof (int) * pSmsHeader->nLoopRecords + */
    /*      sizeof (sfloat) * pSmsHeader->nSpecEnvelopePoints + */
    /*      sizeof(char) * pSmsHeader->nTextCharacters; */
    iVariableSize = sizeof(char) * pSmsHeader->nTextCharacters;

    pSmsHeader->iHeadBSize = sizeof(SMS_Header) + iVariableSize;

    /* write header */
    if (fwrite((void *)pSmsHeader, (size_t)1, (size_t)sizeof(SMS_Header),
                *ppSmsFile) < (size_t)sizeof(SMS_Header))
    {
        sms_error("cannot write output file");
        return(-1);
    }   
    /* write variable part of header */
    if (pSmsHeader->nTextCharacters > 0)
    {
        char *pChStart = (char *) pSmsHeader->pChTextCharacters;
        int iSize = sizeof(char) * pSmsHeader->nTextCharacters;

        if (fwrite ((void *)pChStart, (size_t)1, (size_t)iSize, *ppSmsFile) < 
                (size_t)iSize)
        {
            sms_error("cannot write output file (nTextCharacters)");
            return(-1);
        }   
    }
    return (0);
}

/*! \brief rewrite SMS header and close file
 *
 * \param pSmsFile       pointer to SMS file
 * \param pSmsHeader pointer to header for SMS file
 * \return error code \see SMS_WRERR in SMS_ERRORS 
 */
int sms_writeFile (FILE *pSmsFile, SMS_Header *pSmsHeader)
{
    int iVariableSize;

    rewind(pSmsFile);

    /* check variable size of header */
    iVariableSize = sizeof(char) * pSmsHeader->nTextCharacters;

    pSmsHeader->iHeadBSize = sizeof(SMS_Header) + iVariableSize;

    /* write header */
    if (fwrite((void *)pSmsHeader, (size_t)1, (size_t)sizeof(SMS_Header),
                pSmsFile) < (size_t)sizeof(SMS_Header))
    {
        sms_error("cannot write output file (header)");
        return(-1);
    }   

    if (pSmsHeader->nTextCharacters > 0)
    {
        char *pChStart = (char *) pSmsHeader->pChTextCharacters;
        int iSize = sizeof(char) * pSmsHeader->nTextCharacters;

        if (fwrite ((void *)pChStart, (size_t)1, (size_t)iSize, pSmsFile) < 
                (size_t)iSize)
        {
            sms_error("cannot write output file (nTextCharacters)");
            return(-1);
        }   
    }

    fclose(pSmsFile);
    return (0);
}

/*! \brief write SMS frame
 *
 * \param pSmsFile          pointer to SMS file
 * \param pSmsHeader  pointer to SMS header
 * \param pSmsFrame   pointer to SMS data frame
 * \return 0 on success, -1 on failure
 */
int sms_writeFrame (FILE *pSmsFile, SMS_Header *pSmsHeader,
        SMS_Data *pSmsFrame)
{
    if (fwrite ((void *)pSmsFrame->pSmsData, 1, pSmsHeader->iFrameBSize,
                pSmsFile) < (unsigned int) pSmsHeader->iFrameBSize)
    {
        sms_error("cannot write frame to output file");
        return(-1);
    }
    else return (0);
}


/*! \brief get the size in bytes of the frame in a SMS file 
 *
 * \param pSmsHeader    pointer to SMS header
 * \return the size in bytes of the frame
 */
int sms_frameSizeB (SMS_Header *pSmsHeader)
{
    int iSize, nDet;

    if (pSmsHeader->iFormat == SMS_FORMAT_H ||
            pSmsHeader->iFormat == SMS_FORMAT_IH)
        nDet = 2;/* freq, mag */
    else nDet = 3; /* freq, mag, phase */

    iSize = sizeof (sfloat) * (nDet * pSmsHeader->nTracks);

    if(pSmsHeader->iStochasticType == SMS_STOC_APPROX)
    {       /* stocCoeff + 1 (gain) */
        iSize += sizeof(sfloat) * (pSmsHeader->nStochasticCoeff + 1);
    }
    else if(pSmsHeader->iStochasticType == SMS_STOC_IFFT)
    {
        /* sizeFFT*2 + 1 (gain) */
        iSize += sizeof(sfloat) * (pSmsHeader->nStochasticCoeff * 2 + 1);
    }
    iSize += sizeof(sfloat) * pSmsHeader->nEnvCoeff;
    return(iSize);
}        


/*! \brief function to read SMS header
 *
 * \param pChFileName             file name for SMS file
 * \param ppSmsHeader   (double pointer to) SMS header
 * \param ppSmsFile        (double pointer to) inputfile
 * \return error code \see SMS_ERRORS
 */
int sms_getHeader (char *pChFileName, SMS_Header **ppSmsHeader,
        FILE **ppSmsFile)
{
    int iHeadBSize, iFrameBSize, nFrames;
    int iMagicNumber;

    /* open file for reading */
    if ((*ppSmsFile = fopen (pChFileName, "r")) == NULL)
    {
        sms_error("could not open SMS header");
        return (-1);
    }
    /* read magic number */
    if (fread ((void *) &iMagicNumber, (size_t) sizeof(int), (size_t)1, 
                *ppSmsFile) < (size_t)1)
    {
        sms_error("could not read SMS header");
        return (-1);
    }

    if (iMagicNumber != SMS_MAGIC)
    {
        sms_error("not an SMS file");
        return (-1);
    }

    /* read size of of header */
    if (fread ((void *) &iHeadBSize, (size_t) sizeof(int), (size_t)1, 
                *ppSmsFile) < (size_t)1)
    {
        sms_error("could not read SMS header (iHeadBSize)");
        return (-1);
    }

    if (iHeadBSize <= 0)
    {
        sms_error("bad SMS header size");
        return (-1);
    }

    /* read number of data Frames */
    if (fread ((void *) &nFrames, (size_t) sizeof(int), (size_t)1, 
                *ppSmsFile) < (size_t)1)
    {
        sms_error("could not read SMS number of frames");
        return (-1);
    }

    if (nFrames <= 0)
    {
        sms_error("number of frames <= 0");
        return (-1);
    }

    /* read size of data Frames */
    if (fread ((void *) &iFrameBSize, (size_t) sizeof(int), (size_t)1, 
                *ppSmsFile) < (size_t)1)
    {
        sms_error("could not read size of SMS data");
        return (-1);
    }

    if (iFrameBSize <= 0)
    {
        sms_error("size bytes of frames <= 0");
        return (-1);
    }

    /* allocate memory for header */
    if (((*ppSmsHeader) = (SMS_Header *)malloc (iHeadBSize)) == NULL)
    {
        sms_error("cannot allocate memory for header");
        return (-1);
    }

    /* read header */
    rewind (*ppSmsFile);
    if (fread ((void *) (*ppSmsHeader), 1, iHeadBSize, *ppSmsFile) < (unsigned int) iHeadBSize)
    {
        sms_error("cannot read header of SMS file");
        return (-1);
    }

    /* set pointers to variable part of header */
    /*  if ((*ppSmsHeader)->nLoopRecords > 0) */
    /*      (*ppSmsHeader)->pILoopRecords = (int *) ((char *)(*ppSmsHeader) +  */
    /*          sizeof(SMS_Header)); */

    /*  if ((*ppSmsHeader)->nSpecEnvelopePoints > 0) */
    /*      (*ppSmsHeader)->pFSpectralEnvelope =  */
    /*          (sfloat *) ((char *)(*ppSmsHeader) + sizeof(SMS_Header) +  */
    /*                     sizeof(int) * (*ppSmsHeader)->nLoopRecords); */

    /*  if ((*ppSmsHeader)->nTextCharacters > 0) */
    /*      (*ppSmsHeader)->pChTextCharacters =  */
    /*          (char *) ((char *)(*ppSmsHeader) + sizeof(SMS_Header) +  */
    /*          sizeof(int) * (*ppSmsHeader)->nLoopRecords + */
    /*          sizeof(sfloat) * (*ppSmsHeader)->nSpecEnvelopePoints); */
    if ((*ppSmsHeader)->nTextCharacters > 0)
        (*ppSmsHeader)->pChTextCharacters = (char *)(*ppSmsHeader) + sizeof(SMS_Header);

    return (0);         
}

/*! \brief read an SMS data frame
 *
 * \param pSmsFile         pointer to SMS file
 * \param pSmsHeader       pointer to SMS header
 * \param iFrame               frame number
 * \param pSmsFrame       pointer to SMS frame
 * \return  0 on sucess, -1 on error
 */
int sms_getFrame (FILE *pSmsFile, SMS_Header *pSmsHeader, int iFrame,
        SMS_Data *pSmsFrame)
{    
    if (fseek (pSmsFile, pSmsHeader->iHeadBSize + iFrame * 
                pSmsHeader->iFrameBSize, SEEK_SET) < 0)
    {
        sms_error ("cannot seek to the SMS frame");
        return (-1);
    }
    if ((pSmsHeader->iFrameBSize = 
                fread ((void *)pSmsFrame->pSmsData, (size_t)1, 
                    (size_t)pSmsHeader->iFrameBSize, pSmsFile))
            != pSmsHeader->iFrameBSize)
    {
        sms_error ("cannot read SMS frame");
        return (-1);
    }
    return (0);         
}

/*! \brief  allocate memory for a frame of SMS data
 *
 * \param pSmsFrame      pointer to a frame of SMS data
 * \param nTracks             number of sinusoidal tracks in frame
 * \param nStochCoeff             number of stochastic coefficients in frame
 * \param iPhase              whether phase information is in the frame
 * \param stochType           stochastic resynthesis type
 * \param nStochCoeff             number of envelope coefficients in frame
 * \param nEnvCoeff           number of envelope coefficients in frame
 * \return  0 on success, -1 on error
 */
int sms_allocFrame (SMS_Data *pSmsFrame, int nTracks, int nStochCoeff, int iPhase,
        int stochType, int nEnvCoeff)
{
    sfloat *dataPos;  /* a marker to locate specific data witin smsData */
    /* calculate size of frame */
    int sizeData = 2 * nTracks * sizeof(sfloat);
    sizeData += 1 * sizeof(sfloat); //adding one for nSamples
    if (iPhase > 0) sizeData += nTracks * sizeof(sfloat);
    if (stochType == SMS_STOC_APPROX)
        sizeData += (nStochCoeff + 1) * sizeof(sfloat);
    else if (stochType == SMS_STOC_IFFT)
        sizeData += (2*nStochCoeff + 1) * sizeof(sfloat);
    sizeData += nEnvCoeff * sizeof(sfloat); /* add in number of envelope coefficients (cep or fbins) if any */
    /* allocate memory for data */
    if ((pSmsFrame->pSmsData = (sfloat *) malloc (sizeData)) == NULL)
    {
        sms_error("cannot allocate memory for SMS frame data");
        return (-1);
    }

    /* set the variables in the structure */
    /* \todo why not set these in init functions, then allocate with them?? */
    pSmsFrame->sizeData = sizeData; 
    pSmsFrame->nTracks = nTracks;
    pSmsFrame->nCoeff = nStochCoeff;
    pSmsFrame->nEnvCoeff = nEnvCoeff; 
    /* set pointers to data types within smsData array */
    pSmsFrame->pFSinFreq = pSmsFrame->pSmsData;  
    dataPos =  (sfloat *)(pSmsFrame->pFSinFreq + nTracks);
    memset(pSmsFrame->pFSinFreq, 0, sizeof(sfloat) * nTracks);

    pSmsFrame->pFSinAmp = dataPos;
    dataPos = (sfloat *)(pSmsFrame->pFSinAmp + nTracks);
    memset(pSmsFrame->pFSinAmp, 0, sizeof(sfloat) * nTracks);

    if (iPhase > 0)
    {
        pSmsFrame->pFSinPha = dataPos;
        dataPos = (sfloat *) (pSmsFrame->pFSinPha + nTracks);
        memset(pSmsFrame->pFSinPha, 0, sizeof(sfloat) * nTracks);
    }   
    else    pSmsFrame->pFSinPha = NULL;

    if (stochType == SMS_STOC_APPROX)
    {
        pSmsFrame->pFStocCoeff = dataPos;
        dataPos = (sfloat *) (pSmsFrame->pFStocCoeff + nStochCoeff);
        memset(pSmsFrame->pFStocCoeff, 0, sizeof(sfloat) * nStochCoeff);

        pSmsFrame->pFStocGain = dataPos; 
        dataPos = (sfloat *) (pSmsFrame->pFStocGain + 1);
    }
    else if (stochType == SMS_STOC_IFFT)
    {
        pSmsFrame->pFStocCoeff = dataPos;
        dataPos = (sfloat *) (pSmsFrame->pFStocCoeff + nStochCoeff);
        pSmsFrame->pResPhase = dataPos;
        dataPos = (sfloat *) (pSmsFrame->pResPhase + nStochCoeff);
        pSmsFrame->pFStocGain = dataPos; 
        dataPos = (sfloat *) (pSmsFrame->pFStocGain + 1);
    }
    else
    {
        pSmsFrame->pFStocCoeff = NULL;
        pSmsFrame->pResPhase = NULL;
        pSmsFrame->pFStocGain = NULL;
    }
    if (nEnvCoeff > 0)
        pSmsFrame->pSpecEnv = dataPos;
    else
        pSmsFrame->pSpecEnv = NULL;
    return (0);         
}

/*! \brief  function to allocate an SMS data frame using an SMS_Header
 *
 * this one is used when you have only read the header, such as after 
 * opening a file.
 *
 * \param pSmsHeader       pointer to SMS header
 * \param pSmsFrame     pointer to SMS frame
 * \return  0 on success, -1 on error
 */
int sms_allocFrameH (SMS_Header *pSmsHeader, SMS_Data *pSmsFrame)
{
    int iPhase = (pSmsHeader->iFormat == SMS_FORMAT_HP ||
            pSmsHeader->iFormat == SMS_FORMAT_IHP) ? 1 : 0;
    return (sms_allocFrame (pSmsFrame, pSmsHeader->nTracks, 
                pSmsHeader->nStochasticCoeff, iPhase,
                pSmsHeader->iStochasticType, pSmsHeader->nEnvCoeff));
}


/*! \brief free the SMS data structure
 * 
 * \param pSmsFrame pointer to frame of SMS data
 */
void sms_freeFrame (SMS_Data *pSmsFrame)
{
    free(pSmsFrame->pSmsData);
    pSmsFrame->nTracks = 0;
    pSmsFrame->nCoeff = 0;
    pSmsFrame->sizeData = 0;
    pSmsFrame->pFSinFreq = NULL;
    pSmsFrame->pFSinAmp = NULL;
    pSmsFrame->pFStocCoeff = NULL;
    pSmsFrame->pResPhase = NULL;
    pSmsFrame->pFStocGain = NULL;
}

/*! \brief clear the SMS data structure
 * 
 * \param pSmsFrame pointer to frame of SMS data
 */
void sms_clearFrame (SMS_Data *pSmsFrame)
{
    memset ((char *) pSmsFrame->pSmsData, 0, pSmsFrame->sizeData);
}

/*! \brief copy a frame of SMS_Data 
 *
 * \param pCopySmsData  copy of frame
 * \param pOriginalSmsData  original frame
 *
 */
void sms_copyFrame (SMS_Data *pCopySmsData, SMS_Data *pOriginalSmsData)
{
    /* if the two frames are the same size just copy data */
    if (pCopySmsData->sizeData == pOriginalSmsData->sizeData &&
            pCopySmsData->nTracks == pOriginalSmsData->nTracks)
    {
        memcpy ((char *)pCopySmsData->pSmsData, 
                (char *)pOriginalSmsData->pSmsData,
                pCopySmsData->sizeData);
    }
    /* if frames is different size copy the smallest */
    else
    {   
        int nTracks = MIN (pCopySmsData->nTracks, pOriginalSmsData->nTracks);
        int nCoeff = MIN (pCopySmsData->nCoeff, pOriginalSmsData->nCoeff);

        pCopySmsData->nTracks = nTracks;
        pCopySmsData->nCoeff = nCoeff;
        memcpy ((char *)pCopySmsData->pFSinFreq, 
                (char *)pOriginalSmsData->pFSinFreq,
                sizeof(sfloat) * nTracks);
        memcpy ((char *)pCopySmsData->pFSinAmp, 
                (char *)pOriginalSmsData->pFSinAmp,
                sizeof(sfloat) * nTracks);
        if (pOriginalSmsData->pFSinPha != NULL &&
                pCopySmsData->pFSinPha != NULL)
            memcpy ((char *)pCopySmsData->pFSinPha, 
                    (char *)pOriginalSmsData->pFSinPha,
                    sizeof(sfloat) * nTracks);
        if (pOriginalSmsData->pFStocCoeff != NULL &&
                pCopySmsData->pFStocCoeff != NULL)
        {
            if (pOriginalSmsData->pResPhase != NULL &&
                    pCopySmsData->pResPhase != NULL)
                memcpy ((char *)pCopySmsData->pResPhase, 
                        (char *)pOriginalSmsData->pResPhase,
                        sizeof(sfloat) * nCoeff);
        }
        if (pOriginalSmsData->pFStocGain != NULL &&
                pCopySmsData->pFStocGain != NULL)
            memcpy ((char *)pCopySmsData->pFStocGain, 
                    (char *)pOriginalSmsData->pFStocGain,
                    sizeof(sfloat));
    }
}

/*! \brief function to interpolate two SMS frames
 *
 * this assumes that the two frames are of the same size
 *
 * \param pSmsFrame1            sms frame 1
 * \param pSmsFrame2            sms frame 2
 * \param pSmsFrameOut        sms output frame
 * \param fInterpFactor              interpolation factor
 */
void sms_interpolateFrames (SMS_Data *pSmsFrame1, SMS_Data *pSmsFrame2,
        SMS_Data *pSmsFrameOut, sfloat fInterpFactor)
{
    int i;
    sfloat fFreq1, fFreq2;

    /* interpolate the deterministic part */
    for (i = 0; i < pSmsFrame1->nTracks; i++)
    {
        fFreq1 = pSmsFrame1->pFSinFreq[i];
        fFreq2 = pSmsFrame2->pFSinFreq[i];
        if (fFreq1 == 0) fFreq1 = fFreq2;
        if (fFreq2 == 0) fFreq2 = fFreq1;
        pSmsFrameOut->pFSinFreq[i] = 
            fFreq1 + fInterpFactor * (fFreq2 - fFreq1);
        pSmsFrameOut->pFSinAmp[i] = 
            pSmsFrame1->pFSinAmp[i] + fInterpFactor * 
            (pSmsFrame2->pFSinAmp[i] - pSmsFrame1->pFSinAmp[i]);
    }

    /* interpolate the stochastic part. The pointer is non-null when the frame contains
       stochastic coefficients */
    if (pSmsFrameOut->pFStocGain)
    {
        *(pSmsFrameOut->pFStocGain) = 
            *(pSmsFrame1->pFStocGain) + fInterpFactor *
            (*(pSmsFrame2->pFStocGain) - *(pSmsFrame1->pFStocGain));
    }
    /*! \todo how to interpolate residual phase spectrum */
    for (i = 0; i < pSmsFrame1->nCoeff; i++)
        pSmsFrameOut->pFStocCoeff[i] = 
            pSmsFrame1->pFStocCoeff[i] + fInterpFactor * 
            (pSmsFrame2->pFStocCoeff[i] - pSmsFrame1->pFStocCoeff[i]);

    /* DO NEXT: interpolate spec env here if fbins */
    for (i = 0; i < pSmsFrame1->nEnvCoeff; i++)
        pSmsFrameOut->pSpecEnv[i] = 
            pSmsFrame1->pSpecEnv[i] + fInterpFactor * 
            (pSmsFrame2->pSpecEnv[i] - pSmsFrame1->pSpecEnv[i]);


}
