import ctypes
import numpy as np


class Modem:
    def __init__(self):
        self.clock = 0
        self.rate = 44100
        self.chunk = 1024
        self.t = np.arange(0, self.chunk / self.rate, 1 / float(self.rate), dtype=np.float)
        self.Fdev = 200
        self.Fcs = [500, 1150, 1800, 2500, 3300, 4500, 5400, 6400]
        self.Fc_clock = 8100

    @staticmethod
    def binary(value):
        return format(ctypes.c_uint.from_buffer(ctypes.c_float(value)).value, '#034b')[2:]

    def dispatch(self, data):
        global clock
        sound = []
        for value in data:
            subsound = []
            for idx, bit in enumerate(value):
                subsound.append(self.tone(bit, self.Fcs[idx % 8]))
                if idx % 8 == 7:
                    subsound.append(self.tone(str(int(self.clock)), self.Fc_clock))
                    self.clock = not self.clock
                    sound.append(np.asarray(subsound).sum(axis=0))
                    subsound = []
        sound = np.asarray(sound).ravel()
        return sound

    def tone(self, bit, freq):
        fact = freq + self.Fdev if bit == "1" else freq - self.Fdev
        m = fact * np.ones(self.t.size)
        return 1200 * np.cos(2 * np.pi * np.multiply(m, self.t))

    def convert_data_to_audio(self, data):
        bin_repr = []
        for val in data:
            bin_repr.append(Modem.binary(val))
        return self.dispatch(bin_repr)

    def get_peak(self, hertz):
        return int(round((float(hertz) / self.rate) * self.chunk))

    @staticmethod
    def castfloat(value):
        return ctypes.c_float.from_buffer(ctypes.c_uint(int(value, base=2))).value

    def convert_audio_to_floats(self, audio):
        bitstream = []
        for i in range(0, len(audio), 1024):
            frame = audio[i:i + 1024]
            fft = np.abs(np.fft.rfft(frame))
            """
            from matplotlib import pyplot as plt
            plt.plot(fft[0:250])
            plt.show()
            """
            for Fc in self.Fcs:
                bit = fft[self.get_peak(Fc - self.Fdev)] < fft[self.get_peak(Fc + self.Fdev)]
                bitstream.append(str(int(bit)))

        bitstream = "".join(bitstream)

        floats = []
        for i in range(0, len(bitstream), 32):
            # print(bitstream[i:i+32])
            floats.append(Modem.castfloat(bitstream[i:i + 32]))
            # print(floats[-1])
        return np.asarray(floats)



if __name__ == '__main__':

    np.random.seed(314159265)
    data = np.random.rand(11315) - 0.5

    data = data.astype(np.float32)

    modem = Modem()

    audio_out = modem.convert_data_to_audio(data)

    floats = modem.convert_audio_to_floats(audio_out)
    print(np.allclose(floats, data))