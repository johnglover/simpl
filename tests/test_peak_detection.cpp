#include "test_peak_detection.h"

using namespace simpl;

// ---------------------------------------------------------------------------
//	TestMQPeakDetection
// ---------------------------------------------------------------------------
void TestMQPeakDetection::setUp() {
    _sf = SndfileHandle(TEST_AUDIO_FILE);

    if(_sf.error() > 0) {
        throw Exception(std::string("Could not open audio file: ") +
                        std::string(TEST_AUDIO_FILE));
    }
}

void TestMQPeakDetection::test_find_peaks_in_frame_basic() {
    int frame_size = 2048;

    _pd.clear();
    _pd.frame_size(frame_size);

    Frame f = Frame(frame_size, true);
    Peaks p = _pd.find_peaks_in_frame(&f);
    CPPUNIT_ASSERT(p.size() == 0);
}

void TestMQPeakDetection::test_find_peaks_basic() {
    int frame_size = 512;
    std::vector<sample> audio(frame_size * 2, 0.0);

    _pd.clear();
    _pd.frame_size(frame_size);

    Frames frames = _pd.find_peaks(audio.size(), &audio[0]);

    CPPUNIT_ASSERT(frames.size() == 2);
    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() == 0);
    }
}

void TestMQPeakDetection::test_find_peaks_audio() {
    int num_frames = 5;
    int num_samples = _pd.frame_size() + (_pd.hop_size() * num_frames);

    std::vector<sample> audio(_sf.frames(), 0.0);
    _sf.read(&audio[0], (int)_sf.frames());

    _pd.clear();
    Frames frames = _pd.find_peaks(num_samples,
                                   &(audio[(int)_sf.frames() / 2]));
    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
    }
}

void TestMQPeakDetection::test_find_peaks_change_hop_frame_size() {
    int num_samples = 1024;
    std::vector<sample> audio(num_samples, 0.0);

    _pd.clear();
    _pd.frame_size(256);
    _pd.hop_size(256);

    Frames frames = _pd.find_peaks(num_samples, &audio[0]);
    CPPUNIT_ASSERT(frames.size() == 4);
    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() == 0);
    }
}


// ---------------------------------------------------------------------------
//	TestLorisPeakDetection
// ---------------------------------------------------------------------------
void TestLorisPeakDetection::setUp() {
    _sf = SndfileHandle(TEST_AUDIO_FILE);

    if(_sf.error() > 0) {
        throw Exception(std::string("Could not open audio file: ") +
                        std::string(TEST_AUDIO_FILE));
    }
}

void TestLorisPeakDetection::test_find_peaks_in_frame_basic() {
    int frame_size = 2048;

    _pd.clear();
    _pd.frame_size(frame_size);

    Frame f = Frame(frame_size, true);
    Peaks p = _pd.find_peaks_in_frame(&f);
    CPPUNIT_ASSERT(p.size() == 0);
}

void TestLorisPeakDetection::test_find_peaks_basic() {
    int frame_size = 512;
    std::vector<sample> audio(frame_size * 2, 0.0);

    _pd.clear();
    _pd.frame_size(frame_size);

    Frames frames = _pd.find_peaks(audio.size(), &audio[0]);

    CPPUNIT_ASSERT(frames.size() == 2);
    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() == 0);
    }
}

void TestLorisPeakDetection::test_find_peaks_audio() {
    int num_frames = 5;
    int num_samples = _pd.frame_size() + (_pd.hop_size() * num_frames);

    std::vector<sample> audio(_sf.frames(), 0.0);
    _sf.read(&audio[0], (int)_sf.frames());

    _pd.clear();
    Frames frames = _pd.find_peaks(num_samples,
                                   &(audio[(int)_sf.frames() / 2]));
    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() > 0);
    }
}

void TestLorisPeakDetection::test_find_peaks_change_hop_frame_size() {
    int num_samples = 1024;
    std::vector<sample> audio(num_samples, 0.0);

    _pd.clear();
    _pd.frame_size(256);
    _pd.hop_size(256);

    Frames frames = _pd.find_peaks(num_samples, &audio[0]);
    CPPUNIT_ASSERT(frames.size() == 4);
    for(int i = 0; i < frames.size(); i++) {
        CPPUNIT_ASSERT(frames[i]->num_peaks() == 0);
    }
}
