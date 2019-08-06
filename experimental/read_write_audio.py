#!usr/bin/env python
#coding=utf-8

import pyaudio
import wave
import numpy as np
import math
import struct

def main():
    # define stream chunk
    chunk_size = 1024

    # open a wav format music
    f = wave.open(r"sound_files/test.wav", "rb")
    print(f.getparams())
    # instantiate PyAudio
    p = pyaudio.PyAudio()
    # open stream
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),
                    channels=f.getnchannels(),
                    rate=f.getframerate(),
                    output=True)
    # read data
    data = f.readframes(chunk_size)

    # Read and play audio from wav
    while data:
        sample = np.frombuffer(data, dtype=np.float32)
        sample = sample * 0.3

        stream.write(sample.tobytes())
        data = f.readframes(chunk_size)

        # stop stream

    """
    sampleRate = f.getframerate()  # hertz
    duration = 5.0  # seconds
    frequency = 440.0  # hertz
    audio_time = np.arange(0, duration * sampleRate, dtype=np.float32)
    audio_val = np.cos(frequency * np.pi * audio_time / float(sampleRate))
    for i_chunk in range(int(len(audio_time)/chunk_size)):
        chunk_idx = i_chunk * chunk_size
        stream.write(audio_val[chunk_idx:chunk_idx+1024])
    """

    stream.stop_stream()
    stream.close()

    # close PyAudio
    p.terminate()


if __name__ == '__main__':
    main()