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
/*! \file sms.c
 * \brief initialization, free, and debug functions
 */

#include "sms.h"
#include "SFMT.h" /*!< mersenne twister random number genorator */

char *pChDebugFile = "debug.txt"; /*!< debug text file */
FILE *pDebug; /*!< pointer to debug file */

static char error_message[256];
static int error_status = 0;
static sfloat mag_thresh = .00001; /*!< magnitude threshold for db conversion (-100db)*/
static sfloat inv_mag_thresh = 100000.; /*!< inv(.00001) */
static int initIsDone = 0; /* \todo is this variable necessary? */ 

#define SIZE_TABLES 4096

#define HALF_MAX 1073741823.5  /*!< half the max of a 32-bit word */
#define INV_HALF_MAX (1.0 / HALF_MAX)
#define TWENTY_OVER_LOG10 (20. / LOG10)

/*! \brief initialize global data
 *
 * Currently, just generating the sine and sinc tables.
 * This is necessary before both analysis and synthesis.
 *
 * If using the Mersenne Twister algorithm for random number
 * generation, initialize (seed) it.
 *
 * \return error code \see SMS_MALLOC or SMS_OK in SMS_ERRORS
 */
int sms_init( void )
{
    int iError;
    if (!initIsDone)
    {
        initIsDone = 1;
        if(sms_prepSine (SIZE_TABLES))
        {
            sms_error("cannot allocate memory for sine table");
            return (-1);
        }
        if(sms_prepSinc (SIZE_TABLES))
        {
            sms_error("cannot allocate memory for sinc table");
            return (-1);
        }
    }

#ifdef MERSENNE_TWISTER
    init_gen_rand(1234);
#endif

    return (0);
}

/*! \brief free global data
 *
 * deallocates memory allocated to global arrays (windows and tables)
 */
void sms_free( void )
{
    initIsDone = 0;
    sms_clearSine();
    sms_clearSinc();
}

/*! \brief give default values to an SMS_AnalParams struct 
 * 
 * This will initialize an SMS_AnalParams with values that work
 * for common analyses.  It is useful to start with and then
 * adjust the parameters manually to fit a particular sound
 *
 * Certain things are hard coded in here that will have to 
 * be updated later (i.e. samplerate), so it is best to call this
 * function first, then fill whatever parameters need to be 
 * adjusted.
 * 
 * \param pAnalParams    pointer to analysis data structure
 */
void sms_initAnalParams(SMS_AnalParams *pAnalParams)
{
    pAnalParams->iDebugMode = 0;
    pAnalParams->iFormat = SMS_FORMAT_H;
    pAnalParams->iSoundType = SMS_SOUND_TYPE_MELODY;
    pAnalParams->iStochasticType =SMS_STOC_APPROX;
    pAnalParams->iFrameRate = 300;
    pAnalParams->nStochasticCoeff = 128;
    pAnalParams->fLowestFundamental = 50;
    pAnalParams->fHighestFundamental = 1000;
    pAnalParams->fDefaultFundamental = 100;
    pAnalParams->fPeakContToGuide = .4;
    pAnalParams->fFundContToGuide = .5;
    pAnalParams->fFreqDeviation = .45;
    pAnalParams->iSamplingRate = 44100; /* should be set to the real samplingrate with sms_initAnalysis */
    pAnalParams->iDefaultSizeWindow = 1001;
    pAnalParams->windowSize = 0;
    pAnalParams->sizeHop = 110;
    pAnalParams->fSizeWindow = 3.5;
    pAnalParams->nTracks = 60;
    pAnalParams->maxPeaks = 60;
    pAnalParams->nGuides = 100;
    pAnalParams->iCleanTracks = 1;
    pAnalParams->fMinRefHarmMag = 30;
    pAnalParams->fRefHarmMagDiffFromMax = 30;
    pAnalParams->iRefHarmonic = 1;
    pAnalParams->iMinTrackLength = 40; /*!< depends on iFrameRate normally */
    pAnalParams->iMaxSleepingTime = 40; /*!< depends on iFrameRate normally */
    pAnalParams->fLowestFreq = 50.0;
    pAnalParams->fHighestFreq = 12000.;
    pAnalParams->fMinPeakMag = 0.;
    pAnalParams->iAnalysisDirection = SMS_DIR_FWD;
    pAnalParams->iWindowType = SMS_WIN_BH_70;
    pAnalParams->iSizeSound = 0; /*!< no sound yet */
    pAnalParams->nFrames = 0; /*!< no frames yet */
    pAnalParams->minGoodFrames = 3;
    pAnalParams->maxDeviation = 0.01;
    pAnalParams->analDelay = 100;
    pAnalParams->iMaxDelayFrames = MAX(pAnalParams->iMinTrackLength, pAnalParams->iMaxSleepingTime) + 2 +
        (pAnalParams->minGoodFrames + pAnalParams->analDelay);
    pAnalParams->fResidualAccumPerc = 0.;
    pAnalParams->preEmphasisLastValue = 0.;
    pAnalParams->resetGuides = 1;
    pAnalParams->resetGuideStates = 1;
    /* spectral envelope params */
    pAnalParams->specEnvParams.iType = SMS_ENV_NONE; /* turn off enveloping */
    pAnalParams->specEnvParams.iOrder = 25; /* ... but set default params anyway */
    pAnalParams->specEnvParams.fLambda = 0.00001;
    pAnalParams->specEnvParams.iMaxFreq = 0;
    pAnalParams->specEnvParams.nCoeff = 0;
    pAnalParams->specEnvParams.iAnchor = 0; /* not yet implemented */
}

/*! \brief initialize analysis data structure's arrays
 * 
 *  based on the SMS_AnalParams current settings, this function will
 *  initialize the sound, synth, and fft arrays. It is necessary before analysis.
 *  there can be multple SMS_AnalParams at the same time
 *
 * \param pAnalParams    pointer to analysis paramaters
 * \param pSoundHeader    pointer to sound header
 * \return 0 on success, -1 on error
 */
int sms_initAnalysis(SMS_AnalParams *pAnalParams)
{
    int i;
    SMS_SndBuffer *pSynthBuf = &pAnalParams->synthBuffer;
    SMS_SndBuffer *pSoundBuf = &pAnalParams->soundBuffer;

    /* define the hopsize for each record */
    pAnalParams->sizeHop = (int)(pAnalParams->iSamplingRate /
            (sfloat) pAnalParams->iFrameRate);

    /* define the number of frames and number of samples */
    //  pAnalParams->nFrames = pSoundHeader->nSamples / (sfloat) pAnalParams->sizeHop;
    //  pAnalParams->iSizeSound = pSoundHeader->nSamples;

    /* set the default size window to an odd length */
    pAnalParams->iDefaultSizeWindow = 
        (int)((pAnalParams->iSamplingRate / pAnalParams->fDefaultFundamental) *
                pAnalParams->fSizeWindow / 2) * 2 + 1;

    int sizeBuffer = (pAnalParams->iMaxDelayFrames * pAnalParams->sizeHop) + SMS_MAX_WINDOW;

    /* if storing residual phases, restrict number of stochastic coefficients to the size of the spectrum (sizeHop = 1/2 sizeFft)*/
    if(pAnalParams->iStochasticType == SMS_STOC_IFFT)
        pAnalParams->nStochasticCoeff = sms_power2(pAnalParams->sizeHop);

    /* do the same if spectral envelope is to be stored in frequency bins */
    if(pAnalParams->specEnvParams.iType == SMS_ENV_FBINS)
        pAnalParams->specEnvParams.nCoeff = sms_power2(pAnalParams->specEnvParams.iOrder * 2);
    else if(pAnalParams->specEnvParams.iType == SMS_ENV_CEP)
        pAnalParams->specEnvParams.nCoeff = pAnalParams->specEnvParams.iOrder+1;
    /* if specEnvParams.iMaxFreq is still 0, set it to the same as fHighestFreq (normally what you want)*/
    if(pAnalParams->specEnvParams.iMaxFreq == 0)
        pAnalParams->specEnvParams.iMaxFreq = pAnalParams->fHighestFreq;

    /*\todo this probably doesn't need env coefficients - they aren't getting used */
    sms_allocFrame (&pAnalParams->prevFrame, pAnalParams->nGuides,
            pAnalParams->nStochasticCoeff, 1, pAnalParams->iStochasticType, 0);

    pAnalParams->sizeNextRead = (pAnalParams->iDefaultSizeWindow + 1) * 0.5; /* \todo REMOVE THIS from other files first */

    /* sound buffer */
    if ((pSoundBuf->pFBuffer = (sfloat *) calloc(sizeBuffer, sizeof(sfloat))) == NULL)
    {
        sms_error("could not allocate memory");
        return(-1);
    }
    pSoundBuf->iMarker = -sizeBuffer;
    pSoundBuf->iFirstGood = sizeBuffer;
    pSoundBuf->sizeBuffer = sizeBuffer;

    /* check default fundamental */
    if (pAnalParams->fDefaultFundamental < pAnalParams->fLowestFundamental)
    {
        pAnalParams->fDefaultFundamental = pAnalParams->fLowestFundamental;
    }
    if (pAnalParams->fDefaultFundamental > pAnalParams->fHighestFundamental)
    {
        pAnalParams->fDefaultFundamental = pAnalParams->fHighestFundamental;
    }

    /* initialize peak detection/continuation parameters */
    /*pAnalParams->peakParams.fLowestFreq = pAnalParams->fLowestFundamental;*/
    /*pAnalParams->peakParams.fHighestFreq = pAnalParams->fHighestFreq;*/
    /*pAnalParams->peakParams.fMinPeakMag = pAnalParams->fMinPeakMag;*/
    /*pAnalParams->peakParams.iSamplingRate = pAnalParams->iSamplingRate;*/
    /*pAnalParams->peakParams.iMaxPeaks = SMS_MAX_NPEAKS;*/
    /*pAnalParams->peakParams.fHighestFundamental = pAnalParams->fHighestFundamental;*/
    /*pAnalParams->peakParams.iRefHarmonic = pAnalParams->iRefHarmonic;*/
    /*pAnalParams->peakParams.fMinRefHarmMag = pAnalParams->fMinRefHarmMag;*/
    /*pAnalParams->peakParams.fRefHarmMagDiffFromMax = pAnalParams->fRefHarmMagDiffFromMax;*/
    /*pAnalParams->peakParams.iSoundType = pAnalParams->iSoundType;*/

    /* deterministic synthesis buffer */
    pSynthBuf->sizeBuffer = pAnalParams->sizeHop << 1;
    if((pSynthBuf->pFBuffer = (sfloat *)calloc(pSynthBuf->sizeBuffer, sizeof(sfloat))) == NULL)
    {
        sms_error("could not allocate memory");
        return(-1);
    }
    pSynthBuf->iMarker = -sizeBuffer;
    pSynthBuf->iMarker = pSynthBuf->sizeBuffer;

    /* buffer of analysis frames */
    if ((pAnalParams->pFrames = (SMS_AnalFrame *) calloc(pAnalParams->iMaxDelayFrames, sizeof(SMS_AnalFrame))) == NULL)
    {
        sms_error("could not allocate memory for delay frames");
        return(-1);
    }
    if ((pAnalParams->ppFrames = 
                (SMS_AnalFrame **) calloc(pAnalParams->iMaxDelayFrames, sizeof(SMS_AnalFrame *))) == NULL)
    {
        sms_error("could not allocate memory for pointers to delay frames");
        return(-1);
    }

    /* initialize the frame pointers and allocate memory */
    for (i = 0; i < pAnalParams->iMaxDelayFrames; i++)
    {
        pAnalParams->pFrames[i].iStatus = SMS_FRAME_EMPTY;
        if (((pAnalParams->pFrames[i]).pSpectralPeaks =
                    (SMS_Peak *)calloc (pAnalParams->maxPeaks, sizeof(SMS_Peak))) == NULL)
        {
            sms_error("could not allocate memory for spectral peaks");
            return(-1);
        }
        (pAnalParams->pFrames[i].deterministic).nTracks = pAnalParams->nGuides;
        if (((pAnalParams->pFrames[i].deterministic).pFSinFreq =
                    (sfloat *)calloc (pAnalParams->nGuides, sizeof(sfloat))) == NULL)
        {
            sms_error("could not allocate memory");
            return(-1);
        }
        if (((pAnalParams->pFrames[i].deterministic).pFSinAmp =
                    (sfloat *)calloc (pAnalParams->nGuides, sizeof(sfloat))) == NULL)
        {
            sms_error("could not allocate memory");
            return(-1);
        }
        if (((pAnalParams->pFrames[i].deterministic).pFSinPha =
                    (sfloat *) calloc (pAnalParams->nGuides, sizeof(sfloat))) == NULL)
        {
            sms_error("could not allocate memory");
            return(-1);
        }
        pAnalParams->ppFrames[i] = &pAnalParams->pFrames[i];
    }

    return 0;
}

void sms_changeHopSize(int hopSize, SMS_AnalParams *pAnalParams)
{
    pAnalParams->sizeHop = hopSize;
    pAnalParams->iFrameRate = pAnalParams->iSamplingRate / hopSize;
    int sizeBuffer = (pAnalParams->iMaxDelayFrames * pAnalParams->sizeHop) + SMS_MAX_WINDOW;
    SMS_SndBuffer *pSynthBuf = &pAnalParams->synthBuffer;
    SMS_SndBuffer *pSoundBuf = &pAnalParams->soundBuffer;

    /* if storing residual phases, restrict number of stochastic coefficients to the size of the spectrum (sizeHop = 1/2 sizeFft)*/
    if(pAnalParams->iStochasticType == SMS_STOC_IFFT)
        pAnalParams->nStochasticCoeff = sms_power2(pAnalParams->sizeHop);

    /* sound buffer */
    if ((pSoundBuf->pFBuffer = (sfloat *) calloc(sizeBuffer, sizeof(sfloat))) == NULL)
    {
        sms_error("could not allocate memory");
        return;
    }
    pSoundBuf->iMarker = -sizeBuffer;
    pSoundBuf->iFirstGood = sizeBuffer;
    pSoundBuf->sizeBuffer = sizeBuffer;

    /* deterministic synthesis buffer */
    pSynthBuf->sizeBuffer = pAnalParams->sizeHop << 1;
    if((pSynthBuf->pFBuffer = (sfloat *)calloc(pSynthBuf->sizeBuffer, sizeof(sfloat))) == NULL)
    {
        sms_error("could not allocate memory");
        return;
    }
    pSynthBuf->iMarker = -sizeBuffer;
    pSynthBuf->iMarker = pSynthBuf->sizeBuffer;
}

/*! \brief give default values to an SMS_SynthParams struct 
 * 
 * This will initialize an SMS_SynthParams with values that work
 * for common analyses.  It is useful to start with and then
 * adjust the parameters manually to fit a particular sound
 *
 * \param synthParams    pointer to synthesis parameters data structure
 */
void sms_initSynthParams(SMS_SynthParams *synthParams)
{
    synthParams->iSamplingRate = 44100;
    synthParams->iOriginalSRate = 44100;
    synthParams->iSynthesisType = SMS_STYPE_ALL;
    synthParams->iDetSynthType = SMS_DET_IFFT;
    synthParams->sizeHop = SMS_MIN_SIZE_FRAME;
    synthParams->origSizeHop = SMS_MIN_SIZE_FRAME;
    synthParams->nTracks = 60;
    synthParams->iStochasticType = SMS_STOC_APPROX;
    synthParams->nStochasticCoeff = 128;
    synthParams->deemphasisLastValue = 0;
}

/*! \brief initialize synthesis data structure's arrays
 * 
 *  Initialize the synthesis and fft arrays. It is necessary before synthesis.
 *  there can be multple SMS_SynthParams at the same time
 *  This function also sets some initial values that will create a sane synthesis
 *  environment.
 *
 * This function requires an SMS_Header because it may be called to synthesize
 * a stored .sms file, which contains a header with necessary information.
 *
 * \param pSmsHeader      pointer to SMS_Header
 * \param pSynthParams    pointer to synthesis paramaters
 * \return 0 on success, -1 on error
 */
int sms_initSynth(SMS_SynthParams *pSynthParams)
{
    int sizeHop, sizeFft, err;
    /* set synthesis parameters from arguments and header */
    //  pSynthParams->iOriginalSRate = pSmsHeader->iSamplingRate;
    //  pSynthParams->origSizeHop = pSynthParams->iOriginalSRate / pSmsHeader->iFrameRate;
    //  pSynthParams->iStochasticType = pSmsHeader->iStochasticType;
    //  if(pSynthParams->iSamplingRate <= 0)
    //          pSynthParams->iSamplingRate = pSynthParams->iOriginalSRate;

    /* make sure sizeHop is something to the power of 2 */
    sizeHop = sms_power2(pSynthParams->sizeHop);
    if(sizeHop != pSynthParams->sizeHop)
    {
        sms_error("sizeHop was not a power of two.");
        err = -1;
        pSynthParams->sizeHop = sizeHop;
    }
    sizeFft = sizeHop * 2;

    pSynthParams->pFStocWindow =(sfloat *) calloc(sizeFft, sizeof(sfloat));
    sms_getWindow( sizeFft, pSynthParams->pFStocWindow, SMS_WIN_HANNING );
    pSynthParams->pFDetWindow = (sfloat *) calloc(sizeFft, sizeof(sfloat));
    sms_getWindow( sizeFft, pSynthParams->pFDetWindow, SMS_WIN_IFFT );

    /* allocate memory for analysis data - size of original hopsize */
    /* previous frame to interpolate from */
    /* \todo why is stoch coeff + 1? */
    sms_allocFrame(&pSynthParams->prevFrame, pSynthParams->nTracks,
            pSynthParams->nStochasticCoeff + 1, 1,
            pSynthParams->iStochasticType, 0);

    pSynthParams->pSynthBuff = (sfloat *) calloc(sizeFft, sizeof(sfloat));
    pSynthParams->pMagBuff = (sfloat *) calloc(sizeHop, sizeof(sfloat));
    pSynthParams->pPhaseBuff = (sfloat *) calloc(sizeHop, sizeof(sfloat));
    pSynthParams->pSpectra = (sfloat *) calloc(sizeFft, sizeof(sfloat));

    /* set/check modification parameters */
    //  pSynthParams->modParams.maxFreq = pSmsHeader->iMaxFreq;

    return SMS_OK;
}

int sms_changeSynthHop( SMS_SynthParams *pSynthParams, int sizeHop)
{
    int sizeFft = sizeHop * 2;

    pSynthParams->pSynthBuff = (sfloat *) realloc(pSynthParams->pSynthBuff, sizeFft * sizeof(sfloat));
    pSynthParams->pSpectra = (sfloat *) realloc(pSynthParams->pSpectra, sizeFft * sizeof(sfloat));
    pSynthParams->pMagBuff = (sfloat *) realloc(pSynthParams->pMagBuff, sizeHop * sizeof(sfloat));
    pSynthParams->pPhaseBuff = (sfloat *) realloc(pSynthParams->pPhaseBuff, sizeHop * sizeof(sfloat));
    pSynthParams->pFStocWindow = 
        (sfloat *) realloc(pSynthParams->pFStocWindow, sizeFft * sizeof(sfloat));
    sms_getWindow( sizeFft, pSynthParams->pFStocWindow, SMS_WIN_HANNING );
    pSynthParams->pFDetWindow =
        (sfloat *) realloc(pSynthParams->pFDetWindow, sizeFft * sizeof(sfloat));
    sms_getWindow( sizeFft, pSynthParams->pFDetWindow, SMS_WIN_IFFT );

    pSynthParams->sizeHop = sizeHop;

    return(SMS_OK);
}

/*! \brief free analysis data
 * 
 * frees all the memory allocated to an SMS_AnalParams by
 * sms_initAnalysis
 *
 * \param pAnalParams    pointer to analysis data structure
 */
void sms_freeAnalysis( SMS_AnalParams *pAnalParams )
{
    int i;
    for (i = 0; i < pAnalParams->iMaxDelayFrames; i++)
    {
        free((pAnalParams->pFrames[i]).pSpectralPeaks);
        free((pAnalParams->pFrames[i].deterministic).pFSinFreq);
        free((pAnalParams->pFrames[i].deterministic).pFSinAmp);
        free((pAnalParams->pFrames[i].deterministic).pFSinPha);
    }

    sms_freeFrame(&pAnalParams->prevFrame);
    //        free(pAnalParams->soundBuffer.pFBuffer);
    free(pAnalParams->synthBuffer.pFBuffer);
    free(pAnalParams->pFrames);
    free(pAnalParams->ppFrames);
    //        free(pAnalParams->pFSpectrumWindow);

}

/*! \brief free analysis data
 * 
 * frees all the memory allocated to an SMS_SynthParams by
 * sms_initSynthesis
 *
 * \todo is there a way to make sure the plan has been made
 * already? as it is, it crashes if this is called without one
 * \param pSynthParams    pointer to synthesis data structure
 */
void sms_freeSynth( SMS_SynthParams *pSynthParams )
{
    free(pSynthParams->pFStocWindow);        
    free(pSynthParams->pFDetWindow);
    free (pSynthParams->pSynthBuff);
    free (pSynthParams->pSpectra);
    free (pSynthParams->pMagBuff);
    free (pSynthParams->pPhaseBuff);
    sms_freeFrame(&pSynthParams->prevFrame);

}

/*! \brief set window size for next frame 
 *
 * adjusts the next window size to fit the currently detected fundamental 
 * frequency, or resets to a default window size if unstable.
 *
 * \param iCurrentFrame         number of current frame
 * \param pAnalParams          analysis parameters
 * \return the size of the next window in samples
 */
int sms_sizeNextWindow (int iCurrentFrame, SMS_AnalParams *pAnalParams)
{
    sfloat fFund = pAnalParams->ppFrames[iCurrentFrame]->fFundamental;
    sfloat fPrevFund = pAnalParams->ppFrames[iCurrentFrame-1]->fFundamental;
    int sizeWindow;

    /* if the previous fundamental was stable use it to set the window size */
    if (fPrevFund > 0 && fabs(fPrevFund - fFund) / fFund <= .2)
        sizeWindow = (int) ((pAnalParams->iSamplingRate / fFund) *
                pAnalParams->fSizeWindow * .5) * 2 + 1;
    /* otherwise use the default size window */
    else
        sizeWindow = pAnalParams->iDefaultSizeWindow;

    if (sizeWindow > SMS_MAX_WINDOW)
    {
        fprintf (stderr, "sms_sizeNextWindow error: sizeWindow (%d) too big, set to %d\n", sizeWindow, 
                SMS_MAX_WINDOW);
        sizeWindow = SMS_MAX_WINDOW;
    }

    return sizeWindow;
}

/*! \brief initialize the current frame
 *
 * initializes arrays to zero and sets the correct sample position.
 * Special care is taken at the end the sample source (if there is
 * not enough samples for an entire frame.
 *
 * \param iCurrentFrame            frame number of current frame in buffer
 * \param pAnalParams             analysis parameters
 * \param sizeWindow               size of analysis window 
 * \return -1 on error \todo make this return void
 */
int sms_initFrame(int iCurrentFrame, SMS_AnalParams *pAnalParams, int sizeWindow)
{
    /* clear deterministic data */
    memset ((sfloat *) pAnalParams->ppFrames[iCurrentFrame]->deterministic.pFSinFreq, 0, 
            sizeof(sfloat) * pAnalParams->nGuides);
    memset ((sfloat *) pAnalParams->ppFrames[iCurrentFrame]->deterministic.pFSinAmp, 0, 
            sizeof(sfloat) * pAnalParams->nGuides);
    memset ((sfloat *) pAnalParams->ppFrames[iCurrentFrame]->deterministic.pFSinPha, 0, 
            sizeof(sfloat) * pAnalParams->nGuides);

    /* clear peaks */
    memset ((void *) pAnalParams->ppFrames[iCurrentFrame]->pSpectralPeaks, 0,
            sizeof (SMS_Peak) * pAnalParams->maxPeaks);

    pAnalParams->ppFrames[iCurrentFrame]->nPeaks = 0;
    pAnalParams->ppFrames[iCurrentFrame]->fFundamental = 0;

    pAnalParams->ppFrames[iCurrentFrame]->iFrameNum =  
        pAnalParams->ppFrames[iCurrentFrame - 1]->iFrameNum + 1;
    pAnalParams->ppFrames[iCurrentFrame]->iFrameSize = sizeWindow;

    /* if first frame set center of data around 0 */
    if(pAnalParams->ppFrames[iCurrentFrame]->iFrameNum == 1)
        pAnalParams->ppFrames[iCurrentFrame]->iFrameSample = 0;

    /* if not, increment center of data by sizeHop */
    else
        pAnalParams->ppFrames[iCurrentFrame]->iFrameSample = 
            pAnalParams->ppFrames[iCurrentFrame-1]->iFrameSample + pAnalParams->sizeHop;

    /* check for end of sound */
    if ((pAnalParams->ppFrames[iCurrentFrame]->iFrameSample + (sizeWindow+1)/2) >= pAnalParams->iSizeSound
            && pAnalParams->iSizeSound > 0)
    {
        pAnalParams->ppFrames[iCurrentFrame]->iFrameNum =  -1;
        pAnalParams->ppFrames[iCurrentFrame]->iFrameSize =  0;
        pAnalParams->ppFrames[iCurrentFrame]->iStatus =  SMS_FRAME_END;
    }
    else
    {
        /* good status, ready to start computing */
        pAnalParams->ppFrames[iCurrentFrame]->iStatus = SMS_FRAME_READY;
    }
    return SMS_OK;
}

/*! \brief get deviation from average fundamental
 *\
 * \param pAnalParams             pointer to analysis params
 * \param iCurrentFrame        number of current frame 
 * \return deviation value or -1 if really off
 */
sfloat sms_fundDeviation(SMS_AnalParams *pAnalParams, int iCurrentFrame)
{
    sfloat fFund, fSum = 0, fAverage, fDeviation = 0;
    int i;

    /* get the sum of the past few fundamentals */
    for (i = 0; i < pAnalParams->minGoodFrames; i++)
    {
        fFund = pAnalParams->ppFrames[iCurrentFrame-i]->fFundamental;
        if(fFund <= 0)
            return(-1);
        else
            fSum += fFund;
    }

    /* find the average */
    fAverage = fSum / pAnalParams->minGoodFrames;

    /* get the deviation from the average */
    for (i = 0; i < pAnalParams->minGoodFrames; i++)
        fDeviation += fabs(pAnalParams->ppFrames[iCurrentFrame-i]->fFundamental - fAverage);

    /* return the deviation from the average */
    return (fDeviation / (pAnalParams->minGoodFrames * fAverage));
}


/*! \brief function to create the debug file 
 *
 * \param pAnalParams             pointer to analysis params
 * \return error value \see SMS_ERRORS 
 */
int sms_createDebugFile (SMS_AnalParams *pAnalParams)
{
    if ((pDebug = fopen(pChDebugFile, "w+")) == NULL) 
    {
        fprintf(stderr, "Cannot open debugfile: %s\n", pChDebugFile);
        return(SMS_WRERR);
    }
    else return(SMS_OK);
}

/*! \brief  function to write to the debug file
 *
 * writes three arrays of equal size to a debug text
 * file ("./debug.txt"). There are three arrays for the 
 * frequency, magnitude, phase sets. 
 * 
 * \param pFBuffer1 pointer to array 1
 * \param pFBuffer2 pointer to array 2
 * \param pFBuffer3 pointer to array 3
 * \param sizeBuffer the size of the buffers
 */
void sms_writeDebugData (sfloat *pFBuffer1, sfloat *pFBuffer2, 
        sfloat *pFBuffer3, int sizeBuffer)
{
    int i;
    static int counter = 0;

    for (i = 0; i < sizeBuffer; i++)
        fprintf (pDebug, "%d %d %d %d\n", counter++, (int)pFBuffer1[i],
                (int)pFBuffer2[i], (int)pFBuffer3[i]);

}

/*! \brief  function to write the residual sound file to disk
 *
 * writes the "debug.txt" file to disk and closes the file.
 */
void sms_writeDebugFile ()
{
    fclose (pDebug);
}

/*! \brief convert from magnitude to decibel
 *
 * \param x      magnitude (0:1)
 * \return         decibel (0: -100)
 */
sfloat sms_magToDB( sfloat x)
{
    if(x < mag_thresh)
        return 0.0;
    else
        //return(20. * log10(x * inv_mag_thresh));
        return(TWENTY_OVER_LOG10 * log(x * inv_mag_thresh));
        /*return(TWENTY_OVER_LOG10 * log(x));*/
}

/*! \brief convert from decibel to magnitude
 *
 * \param x     decibel (0-100)
 * \return        magnitude (0-1)
 */
sfloat sms_dBToMag( sfloat x)
{
    if(x < 0.00001)
        return 0.0;
    else
        return(mag_thresh * pow(10., x*0.05));
        /*return pow(10.0, x*0.05);*/
}

/*! \brief convert an array from magnitude to decibel 
 *
 * Depends on a  linear threshold that indicates the bottom end
 * of the dB scale (magnutdes at this value will convert to zero).
 * \see sms_setMagThresh
 *
 * \param sizeArray     size of array
 * \param pArray pointer to array
 */
void sms_arrayMagToDB( int sizeArray, sfloat *pArray)
{
    int i;
    for( i = 0; i < sizeArray; i++)
        pArray[i] = sms_magToDB(pArray[i]);
}

/*! \brief convert and array from decibel (0-100) to magnitude (0-1)
 *
 * depends on the magnitude threshold
 * \see sms_setMagThresh
 *
 * \param sizeArray     size of array
 * \param pArray pointer to array
 */
void sms_arrayDBToMag( int sizeArray, sfloat *pArray)
{
    int i;
    for( i = 0; i < sizeArray; i++)
        pArray[i] = sms_dBToMag(pArray[i]);
}
/*! \brief set the linear magnitude threshold
 *
 * magnitudes below this will go to zero when converted to db.
 * it is limited to 0.00001 (-100db)
 *
 * \param x  threshold value
 */
void sms_setMagThresh( sfloat x)
{
    /* limit threshold to -100db */
    if(x < 0.00001) 
        mag_thresh = 0.00001;
    else
        mag_thresh = x;
    inv_mag_thresh = 1. / mag_thresh;
}

/*! \brief get a string containing information about the error code 
 *
 * \param pErrorMessage pointer to error message string
 */
void sms_error(char *pErrorMessage) {
    strncpy(error_message, pErrorMessage, 256);
    error_status = -1;
}

/*! \brief check if an error has been reported
 *
 * \return  -1 if there is an error, 0 if ok
 */
int sms_errorCheck() 
{
    return(error_status);
}

/*! \brief get a string containing information about the last error 
 *
 * \return  pointer to a char string, or NULL if no error
 */
char* sms_errorString() 
{
    if (error_status)
    {
        error_status = 0;
        return error_message;
    }
    else return NULL;
}

/*! \brief random number genorator
 *
 * \return random number between -1 and 1
 */
sfloat sms_random()
{
#ifdef MERSENNE_TWISTER
    return(genrand_real1()); 
#else
    return((sfloat)(random() * 2 * INV_HALF_MAX));
#endif
}

/*! \brief Root Mean Squared of an array
 *
 * \return RMS energy
 */
sfloat sms_rms(int sizeArray, sfloat *pArray)
{
    int i;
    sfloat mean_squared = 0.;
    for( i = 0; i < sizeArray; i++)
        mean_squared += pArray[i] * pArray[i];

    return(sqrtf(mean_squared / sizeArray));
}

/*! \brief make sure a number is a power of 2
 *
 * \return a power of two integer >= input value
 */
int sms_power2( int n)
{
    int p = -1;
    int N = n;
    while(n)
    {
        n >>= 1;
        p++;
    }

    if(1<<p == N) /* n was a power of 2 */
    {
        return(N); 
    }
    else  /* make the new value larger than n */
    {
        p++;
        return(1<<p);
    }
}

/*! \brief compute a value for scaling frequency based on the well-tempered scale
 *
 * \param x linear frequency value
 * \return (1.059...)^x, where 1.059 is the 12th root of 2 precomputed
 */
sfloat sms_scalarTempered( sfloat x)
{
    return(powf(1.0594630943592953, x));
}

/*! \brief scale an array of linear frequencies to the well-tempered scale
 *
 * \param sizeArray size of the array
 * \param pArray pointer to array of frequencies
 */
void sms_arrayScalarTempered( int sizeArray, sfloat *pArray)
{
    int i;
    for( i = 0; i < sizeArray; i++)
        pArray[i] = sms_scalarTempered(pArray[i]);
}
