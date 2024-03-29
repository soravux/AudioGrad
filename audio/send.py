import sys
import pyaudio
import quietnet
import options
import psk

FORMAT = pyaudio.paInt16
CHANNELS = options.channels
RATE = options.rate
FREQ = options.freq
FREQ_OFF = 0
FRAME_LENGTH = options.frame_length
DATASIZE = options.datasize

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)


def make_buffer_from_bit_pattern(pattern, on_freq, off_freq):
    """ Takes a pattern and returns an audio buffer that encodes that pattern """
    # the key's middle value is the bit's value and the left and right bits are the bits before and after
    # the buffers are enveloped to cleanly blend into each other

    last_bit = pattern[-1]
    output_buffer = []
    offset = 0

    for i in range(len(pattern)):
        bit = pattern[i]
        if i < len(pattern) - 1:
            next_bit = pattern[i+1]
        else:
            next_bit = pattern[0]

        freq = on_freq if bit == '1' else off_freq
        tone = quietnet.tone(freq, DATASIZE, offset=offset)
        output_buffer += quietnet.envelope(tone, left=last_bit=='0', right=next_bit=='0')
        offset += DATASIZE
        last_bit = bit

    return quietnet.pack_buffer(output_buffer)

def play_buffer(buffer):
    output = b''.join(buffer)
    stream.write(output)

if __name__ == "__main__":
    print("Use ctrl-c to exit")

    try:
        # get user input and play message
        while True:
            message = input("> ")
            try:
              pattern1 = psk.encode(message[::2])
              pattern2 = psk.encode(message[1::2])
              buffer1 = make_buffer_from_bit_pattern(pattern1, FREQ, FREQ_OFF)
              buffer2 = make_buffer_from_bit_pattern(pattern2, 800, FREQ_OFF)
              for i in range(min(len(buffer1), len(buffer2))):
                #print(buffer1[i], int.from_bytes(buffer1[i], byteorder="little", signed=True))
                buffer1[i] = (int.from_bytes(buffer1[i], byteorder="little", signed=True)
                            + int.from_bytes(buffer2[i], byteorder="little", signed=True)
                             ).to_bytes(2, byteorder="little", signed=True)
              play_buffer(buffer1)
            except KeyError:
              print("Messages may only contain printable ASCII characters.")
    except KeyboardInterrupt:
        # clean up our streams and exit
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("exited cleanly")
