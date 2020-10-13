"""
This is a simple program that 'visualizes' video material into sound.
It's a total rip-off from 3b1b's video on Hilbert curves:
https://www.youtube.com/watch?v=3s7h2MHQtxc
A couple of functions should be accelerated by your GPU but results may vary.

Author: Koen van Wel
Email: koenvwel@gmail.com
"""

import getopt
import io
import math
import struct
import sys
import time
import threading
from collections import deque
from typing import Iterable, List, Deque
import numba
import numpy as np
import pyaudio
from hilbertcurve.hilbertcurve import HilbertCurve
import cv2

CHANNELS = 1
RATE = int(44100 / 2)
FORMAT = pyaudio.paInt16
MAX_AMPLITUDE = 32767 / 4  # 32767 = paInt16.max_value()
MIN_FREQ = 500
MAX_FREQ = 5000
HILBERT_PRECISION = 5
INVERTED = False
NUMBA_PARALLEL = False
MAX_BUFFER = 10  # Amount of frames the audio stream can fall behind
SHOW_VIDEO = True
VIDEO_MAX_WIDTH = 480
VIDEO_MAX_HEIGHT = 320
SHOW_PLOT = True
FOURIER_TRANSFORM_WAVE = True

_MAX_USED_FREQ = min(MAX_FREQ, int(RATE / 2))
_HILBERT_RESOLUTION = 2 ** HILBERT_PRECISION

if SHOW_PLOT:
    if SHOW_VIDEO:
        import matplotlib as mpl

        mpl.use('Agg')
    import matplotlib.pyplot as plt


def video_to_waves(video: cv2.VideoCapture) -> Iterable[List[int]]:
    """
    Take a video, and yield a sound wave for each frame. Will yield the frame and the 'prepared'
    frame for testing purposes.
    :param video: A cv2.VideoCapture object from which the frames will be yielded
    :return: triple of the audio wave, the frame, and the prepared frame
    """
    fps = video.get(cv2.CAP_PROP_FPS)
    hilbert_curve = HilbertCurve(HILBERT_PRECISION, 2)
    previous = 0
    while video.isOpened():
        start = time.perf_counter()
        if start - previous < 1 / fps:
            time.sleep((1 / fps) - (start - previous))
        ret, frame = video.read()
        if not ret:
            break
        prepared = prepare_frame(frame)
        wave = frame_to_wave(prepared, hilbert_curve, 1 / fps)
        previous = start
        # end = time.perf_counter()
        yield wave, frame, prepared


def prepare_frame(frame: np.ndarray) -> np.ndarray:
    """
    Prepare a frame to be processed
    :param frame: Input frame
    :return: an ndarray with shape (_HILBERT_RESOLUTION, _HILBERT_RESOLUTION)
    """
    gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray_scale, (_HILBERT_RESOLUTION, _HILBERT_RESOLUTION))


def frame_to_wave(frame: np.ndarray, hilbert_curve: HilbertCurve, delta: float) -> List[int]:
    """
    Transform a prepared frame of shape (_HILBERT_RESOLUTION, _HILBERT_RESOLUTION) to a sound wave
    :param frame: The prepared frame
    :param hilbert_curve: The hilbert curve for deciding which frequency each pixel represents
    :param delta: The time of the wave (expected to be 1/fps)
    :return: A list representing an audio wave
    """
    assert frame.shape == (_HILBERT_RESOLUTION, _HILBERT_RESOLUTION)
    vec = frame_to_gain_vector(frame, hilbert_curve)
    return vector_to_wave(numba.typed.List(vec), delta)


def frame_to_gain_vector(frame: np.ndarray, hilbert_curve: HilbertCurve) -> Iterable[int]:
    """
    Create a gain vector from a frame
    :param frame: The prepared frame
    :param hilbert_curve: The hilbert curve for deciding which frequency each pixel represents
    :return: A generator with the gain that each pixel represents in order of frequency
    """
    for i in range(frame.size):
        x, y = hilbert_curve.coordinates_from_distance(i)
        yield frame[y, x] if not INVERTED else 255 - frame[y, x]


@numba.njit(parallel=NUMBA_PARALLEL)
def vector_to_wave(vector: numba.types.float64[:], delta: float) -> np.ndarray:
    """
    Create a waveform based on a vector of gains
    :param vector: vector of gains between 0-255
    :param delta: time of the waveform
    :return: np.ndarray[numba.int64] representing a waveform
    """
    frame_count = int(RATE * delta)
    summed_wave = np.zeros(frame_count, numba.float64)
    for i in numba.prange(len(vector)):
        gain = vector[i]
        frequency = (_MAX_USED_FREQ - MIN_FREQ) / len(vector) * (i + 1) + MIN_FREQ
        amplitude = get_amplitude(gain, frequency)

        this_sine = generate_freq(frequency, frame_count, amplitude)
        summed_wave += this_sine
    summed_wave /= len(vector)
    return summed_wave.astype(numba.int64)


@numba.njit  # Parallelization cannot be helpful
def get_amplitude(gain: int, frequency: float) -> float:
    """
    Not yet implemented
    Get a frequency based on a gain between 0-255 and a frequency. Always under MAX_AMPLITUDE
    :param gain:
    :param frequency:
    :return: an amplitude
    """
    return (gain / 255) * MAX_AMPLITUDE  # TODO create equalizer


@numba.njit(parallel=NUMBA_PARALLEL)
def generate_freq(frequency: float, frame_count: int, amplitude: float) -> np.ndarray:
    """
    Numba function to generate a specific frequency and return it as a np.ndarray
    :param frequency: The frequency that is to be generated
    :param frame_count: The amount of frames to fill
    :param amplitude: The amplitude of the frequency
    :return: np.ndarray[numba.float64] of the values the frequency is made up of
    """
    wave_data = np.zeros(frame_count, numba.float64)
    for i in numba.prange(frame_count):
        frames_per_wave = RATE / frequency
        place_in_wave = i / frames_per_wave
        wave_data[i] = math.sin(place_in_wave * (2 * math.pi)) * amplitude
    return wave_data


def wave_to_data(wave: List[int]) -> bytes:
    """
    Take a list of values in a wave and pack them in a C struct
    :param wave: the audio wave
    :return: C struct representing the wave
    """
    number_of_bytes = str(len(wave))
    return struct.pack(number_of_bytes + 'h', *wave)


def plot_to_ndarray(data) -> np.ndarray:
    """
    Plot a figure to an ndarray
    :param data: data to be plotted
    :return: np.ndarray compatible with cv2 (RGBA)
    """
    fig = plt.figure()
    line = plt.subplot(111)
    line.plot(data)
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw')
    io_buf.seek(0)
    img_arr = np.reshape(
        np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
        newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1)
    )
    io_buf.close()
    plt.close(fig)
    return img_arr


def stream_from_buffer(stream: pyaudio.Stream, buffer: Deque) -> bool:
    """
    Reads buffer and writes to pyaudio stream
    Also shows the video and the plot if SHOW_VIDEO and SHOW_PLOT are True respectively
    :param stream: pyaudio.Stream to write to
    :param buffer: buffer to read from (pops a wave, frame, prepared_frame triplet)
    :return: False if the user closed the window by pressing 'q', True otherwise
    """
    if len(buffer) > 0:
        wave, frame, prepared_frame = buffer.popleft()
        stream.write(wave_to_data(wave))

        if SHOW_VIDEO:
            height, width = frame.shape[1::-1]
            if width > VIDEO_MAX_WIDTH or height > VIDEO_MAX_HEIGHT:
                frame = cv2.resize(frame, (VIDEO_MAX_WIDTH, VIDEO_MAX_HEIGHT))

            resized_prepared_frame = cv2.cvtColor(
                cv2.resize(prepared_frame, frame.shape[1::-1]),
                cv2.COLOR_GRAY2RGB
            )
            concat = cv2.vconcat([frame, resized_prepared_frame])

            if SHOW_PLOT:
                if FOURIER_TRANSFORM_WAVE:
                    wave = np.fft.rfft(wave).real

                plot = cv2.cvtColor(
                    cv2.resize(plot_to_ndarray(wave), concat.shape[1::-1]),
                    cv2.COLOR_RGBA2RGB
                )
                cv2.imshow("Video", cv2.hconcat([concat, plot]))
            else:
                cv2.imshow("Video", concat)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        elif SHOW_PLOT:
            if FOURIER_TRANSFORM_WAVE:
                wave = np.fft.rfft(wave).real
            plt.clf()
            plt.plot(wave)
            plt.pause(0.0001)

    if len(buffer) > MAX_BUFFER:
        for _ in range(int(MAX_BUFFER / 2)):
            buffer.popleft()

    return True


def synesthesia(path: str):
    """
    Read video from a path. Create a thread that changes each frame into a
    waveform and puts that waveform in a buffer
    Read that buffer from the main thread and write it to the audio stream
    :param path: path to video
    """
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
    video = cv2.VideoCapture(path)
    buffer = deque()

    def wave_buffering():
        for result in video_to_waves(video):
            buffer.append(result)

    wave_buffering_thread = threading.Thread(target=wave_buffering)
    wave_buffering_thread.start()

    while stream.is_active() and video.isOpened():
        if not stream_from_buffer(stream, buffer):
            break

    cv2.destroyAllWindows()
    video.release()
    stream.stop_stream()
    stream.close()
    audio.terminate()
    wave_buffering_thread.join()


def main(argv):
    """
    Main function. Reads sys.argv and runs synesthesia function based on the input
    :param argv:
    """
    def usage():
        print("Usage: " + sys.argv[0] + " -i <input path>")

    input_path = None
    try:
        opts, _ = getopt.getopt(argv[1:], "hi:", ["help", "inpath="])
    except getopt.GetoptError as error:
        usage()
        sys.exit(error)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-i", "--inpath"):
            input_path = arg
    if input_path is None:
        usage()
        sys.exit(1)
    synesthesia(input_path)


if __name__ == '__main__':
    main(sys.argv)
