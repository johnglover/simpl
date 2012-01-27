import simpl
import numpy as np

def sine_wave(n, f=220, sr=44100):
    s = simpl.zeros(n)
    for i in range(n):
        s[i] = np.sin(2.0 * np.pi * f * i/sr)
    return s

def noisy_sine_wave(n, f=220, sr=44100):
    s = simpl.zeros(n)
    for i in range(n):
        s[i] = np.sin(2*np.pi*f*i/sr) + (np.random.random() / 4)
    return s

def sinechirpsine():
    initial_freq = 220
    final_freq = 440
    amp = 0.5
    section_length = 1 # seconds
    sampling_rate = 44100
    
    audio = simpl.zeros(section_length*sampling_rate*3)
    chirp_freq = initial_freq
    chirp_rate = (final_freq - initial_freq) / 2
    
    for i in range(section_length*sampling_rate):
        t = float(i) / sampling_rate
        audio[i] = amp * np.sin(2 * np.pi * initial_freq * t)
        audio[i+(section_length*sampling_rate)] = amp * np.sin(2 * np.pi * (initial_freq*t + chirp_rate*t*t))
        audio[i+(section_length*sampling_rate*2)] = amp * np.sin(2 * np.pi * final_freq * t)
    
    return audio
