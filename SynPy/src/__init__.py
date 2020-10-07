"""
This is a simple program that 'visualizes' video material into sound.
It's a total rip-off from 3b1b's video on Hilbert curves: https://www.youtube.com/watch?v=3s7h2MHQtxc
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
import numba
from collections import deque
from typing import Iterable, List

import numpy as np
import cv2
import pyaudio

from hilbertcurve.hilbertcurve import HilbertCurve

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
SHOW_PLOT = False

_MAX_USED_FREQ = min(MAX_FREQ, int(RATE/2))
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
    return vector_to_wave(list(vec), delta)


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
def vector_to_wave(vector: numba.types.float64[:], delta: float):
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
def get_amplitude(gain: int, frequency: float):
    return (gain / 255) * MAX_AMPLITUDE  # TODO create equalizer


@numba.njit(parallel=NUMBA_PARALLEL)
def generate_freq(frequency: float, frame_count: int, amplitude: float):
    wave_data = np.zeros(frame_count, numba.float64)
    for i in numba.prange(frame_count):
        frames_per_wave = RATE / frequency
        place_in_wave = i / frames_per_wave
        wave_data[i] = math.sin(place_in_wave * (2 * math.pi)) * amplitude
    return wave_data


def wave_to_data(wave: List[int]):
    """
    Take a list of values in a wave and pack them in a C struct
    :param wave: the audio wave
    :return: C struct representing the wave
    """
    number_of_bytes = str(len(wave))
    return struct.pack(number_of_bytes + 'h', *wave)


def plot_to_ndarray(data):
    """
    Plot a figure to an ndarray
    :param data: data to be plotted
    :return: np.ndarray compatible with cv2 (RGBA)
    """
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(data)
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


def synesthesia(path):
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
    video = cv2.VideoCapture(path)

    buffer = deque()

    def wave_buffering():
        for result in video_to_waves(video):
            buffer.append(result)

    wave_buffering_thread = threading.Thread(target=wave_buffering)
    wave_buffering_thread.start()
    while stream.is_active() and video.isOpened():
        if len(buffer) > 0:
            wave, frame, prepared = buffer.popleft()
            stream.write(wave_to_data(wave))

            if SHOW_VIDEO:
                resized_prepared = cv2.cvtColor(cv2.resize(prepared, frame.shape[1::-1]),
                                                cv2.COLOR_GRAY2RGB)
                concat = cv2.vconcat([frame, resized_prepared])

                if SHOW_PLOT:
                    plot = cv2.cvtColor(cv2.resize(plot_to_ndarray(np.fft.rfft(wave).real), concat.shape[1::-1]),
                                        cv2.COLOR_RGBA2RGB)
                    cv2.imshow("Video", cv2.hconcat([concat, plot]))
                else:
                    cv2.imshow("Video", concat)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

            elif SHOW_PLOT:
                plt.clf()
                plt.plot(wave)
                plt.pause(0.0001)

        if len(buffer) > MAX_BUFFER:
            halve_buffer = int(MAX_BUFFER / 2)
            # TODO drop half of buffer
    cv2.destroyAllWindows()
    video.release()
    stream.stop_stream()
    stream.close()
    pa.terminate()


def main(argv):
    def usage():
        print("Usage: " + sys.argv[0] + " [-i <input path> -b <buffer size>")
    input_path = None
    try:
        opts, args = getopt.getopt(argv[1:], "hi:", ["help","inpath="])
    except getopt.GetoptError as e:
        usage()
        sys.exit(e)
    for opt, arg in opts:
        if opt == '-h':
            usage()
            sys.exit()
        elif opt in ("-i", "--inpath"):
            input_path = arg
    if input_path is None:
        usage()
        exit(1)
    synesthesia(input_path)


if __name__ == '__main__':
    main(sys.argv)
