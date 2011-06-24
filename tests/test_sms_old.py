# Copyright (c) 2009 John Glover, National University of Ireland, Maynooth
#  
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of 
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, M  02111-1307  USA

import unittest
from pysms import SMS_Header, SMS_Data, SMS_SndHeader, SMS_AnalParams, \
                  sms_openSF, sms_errorString, sms_getSound, \
                  sms_fillHeader, sms_init, sms_initAnalysis, sms_allocFrameH, \
                  sms_freeFrame, sms_freeAnalysis, sms_free
from scipy import zeros, sin, pi, asarray, int16
from scipy.io.wavfile import read, write
import os
import random
from pylab import plot, show

TEST_AUDIO_FILE = "Tests.wav"

# Create a test audio file (1 second of a sine wave at 220 Hz)
test_audio = zeros(44100)
for sample_number in range(test_audio.size):
    test_audio[sample_number] = sin(2 * pi * 220 * sample_number / 44100.0)
# convert to int values
test_audio *= 32767
test_audio = asarray(test_audio, int16)

def create_test_audio_file():
    "Create a test audio file in the current directory"
    write(TEST_AUDIO_FILE, 44100, test_audio)    
    
def delete_test_audio_file():
    "Delete the test audio file created by the function create_test_audio_file"
    os.remove(TEST_AUDIO_FILE)

class TestSoundIO(unittest.TestCase):
    def setUp(self):
        self.snd_header = SMS_SndHeader()
        
    def test_sms_openSF_file_exists(self):
        "sms_openSF returns True when trying to open an existing file"
        create_test_audio_file()
        self.assert_(sms_openSF(TEST_AUDIO_FILE, self.snd_header) == 0)
        delete_test_audio_file()
        
    def test_sms_openSF_file_missing(self):
        "sms_openSF returns False when trying to open a file that doesn't exist"
        file_path = ""
        max_file_names = 1000
        count = 0
        class MaxFilesReached(Exception): pass
        # create a path to a non-existent file
        while True:
            file_path = str(random.randint(0, max_file_names)) + ".wav"
            if not os.path.isfile(file_path):
                break
            if count > max_file_names:
                raise MaxFilesReached
            count += 1
        # call sms_openSF, should return an error
        self.assertRaises(IndexError, sms_openSF, file_path, self.snd_header)
        
    def test_sms_getSound(self):
        "sms_getSound"
        create_test_audio_file()
        self.assert_(sms_openSF(TEST_AUDIO_FILE, self.snd_header) == 0)
        frame_size = 512
        frame = zeros(frame_size).astype('float32')
        self.assert_(sms_getSound(self.snd_header, frame, 0) == 0)
        # test that values read in are the same as those written (allowing for some rounding errors)
        class SampleMismatch(Exception): pass
        for sample_number in range(frame_size):
            if abs((test_audio[sample_number] / 32768.0) - frame[sample_number] > 0.000001):
                raise SampleMismatch
        delete_test_audio_file()
    
class TestInit(unittest.TestCase):
    def setUp(self):
        self.snd_header = SMS_SndHeader()
        self.sms_header = SMS_Header()
        self.data = SMS_Data()
        self.analysis_params = SMS_AnalParams()
        
    def tearDown(self):
        sms_freeFrame(self.data)
        
    def test_sms_fillHeader(self):
        "sms_fillHeader copies data from an SMS_AnalParams to an SMS_Header"
        data_fields = ["nFrames", "iFormat", "iFrameRate", "iStochasticType", \
                       "nTracks", "iSamplingRate", "nStochasticCoeff"]
        sms_fillHeader(self.sms_header, self.analysis_params, "")
        for field in data_fields:
            self.assert_(eval("self.sms_header."+field) == eval("self.analysis_params."+field))
            
    def test_sms_init(self):
        "sms_init"
        self.assert_(sms_init() == 0)
        
    def test_sms_initAnalysis(self):
        "sms_initAnalysis"
        create_test_audio_file()
        if(sms_openSF(TEST_AUDIO_FILE, self.snd_header)):
            raise NameError("error opening sound file: " + sms_errorString())
        self.assert_(sms_initAnalysis(self.analysis_params, self.snd_header) == 0)
        delete_test_audio_file()
        
    def test_sms_allocFrameH(self):
        "sms_allocFrameH"
        create_test_audio_file()
        if(sms_openSF(TEST_AUDIO_FILE, self.snd_header)):
            raise NameError("error opening sound file: " + sms_errorString())
        self.assert_(sms_allocFrameH(self.sms_header, self.data) == 0)
        delete_test_audio_file()

if __name__ == '__main__':
    unittest.main()
