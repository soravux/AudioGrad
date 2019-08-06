import os

import pyaudio
import numpy as np
import pickle as pkl


def open_stream(fs):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)
    return p, stream


def generate_tone(fs, freq, duration):
    npsin = np.sin(2 * np.pi * np.arange(fs * duration)[np.newaxis, :] * freq[:, np.newaxis, ] / fs)
    samples = npsin.astype(np.float32)
    return 0.1 * samples


def run_main():

    gradient_path = "gradients/test"
    epoch_files = [os.path.join(gradient_path, x) for x in os.listdir(gradient_path)]
    epoch_files.sort()

    p = pyaudio.PyAudio()
    fs = 44100
    duration = 0.03
    f = 300.0
    stream = p.open(format=p.get_format_from_width(4),
                    channels=1,
                    rate=fs,
                    output=True)

    for file in epoch_files:
        grads = pkl.load(open(file, "br"))
        # right now only the norm of one layer is used.
        grad = grads["fc1.weight"]
        n_batch = grad.shape[0]

        norm_grad = np.linalg.norm(grad.reshape(n_batch, -1), axis=-1)
        tone = f + (norm_grad * 500.0)
        samples = generate_tone(fs, tone, duration).astype(np.float32)
        # here we write to audio or wav file
        stream.write(samples.flatten().tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()


if __name__ == "__main__":
    run_main()