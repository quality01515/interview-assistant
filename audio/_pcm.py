"""
Helpers: float/int conversion, channel downmix, resampling.

Kept dependency-light: just numpy + scipy.signal.

Live audio arrives in small buffers (1024 frames ≈ 21 ms). Use PcmConverter —
one per stream — for those: it keeps the resampler's filter state and timing
between buffers. Resampling each buffer on its own (to_mono_16k_int16) restarts
the filter every 21 ms, which adds an audible buzz only ~22 dB below the voice
plus a slow timing drift, and noticeably hurts transcription accuracy.
to_mono_16k_int16 is fine for a single self-contained clip.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter, resample_poly


def _decode(raw: bytes, sample_format: str, channels: int) -> np.ndarray:
    if sample_format == "int16":
        arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_format == "int32":
        arr = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sample_format == "float32":
        arr = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported sample format: {sample_format}")
    if channels > 1:
        usable = (arr.size // channels) * channels
        arr = arr[:usable].reshape(-1, channels).mean(axis=1)
    return arr


def _encode(arr: np.ndarray) -> bytes:
    return np.clip(arr * 32767.0, -32768, 32767).astype(np.int16).tobytes()


def to_mono_16k_int16(raw: bytes, *, sample_rate: int, channels: int, sample_format: str, target_rate: int = 16000) -> bytes:
    """
    Convert ONE self-contained PCM clip of arbitrary format/rate/channels to
    16-bit mono `target_rate` Hz. For a continuous stream use PcmConverter.

    `sample_format` is one of: 'int16', 'int32', 'float32'.
    """
    arr = _decode(raw, sample_format, channels)
    if sample_rate != target_rate:
        from math import gcd
        g = gcd(sample_rate, target_rate)
        arr = resample_poly(arr, target_rate // g, sample_rate // g).astype(np.float32)
    return _encode(arr)


class PcmConverter:
    """
    Streaming equivalent of to_mono_16k_int16 for one continuous audio stream.

    Anti-alias low-pass (FIR, filter state carried across buffers), then
    interpolation at the output rate with a fractional read position that also
    carries across buffers — so consecutive buffers join seamlessly and the
    output rate is exact (no drift).
    """

    _TAPS = 129

    def __init__(self, *, sample_rate: int, channels: int, sample_format: str,
                 target_rate: int = 16000):
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self.sample_format = sample_format
        self.target_rate = int(target_rate)
        self._resample = self.sample_rate != self.target_rate
        if self._resample:
            # Pass band up to 45% of the LOWER rate: 7.2 kHz for 16 kHz output,
            # which keeps everything speech recognition uses.
            cutoff = 0.45 * min(self.sample_rate, self.target_rate)
            self._taps = firwin(self._TAPS, cutoff, fs=self.sample_rate)
            self._zi = np.zeros(self._TAPS - 1)
            self._step = self.sample_rate / self.target_rate
            self._pos = 0.0               # next output position, in input samples, relative to this buffer
            self._hist = np.zeros(3)      # last 3 filtered samples of earlier buffers (positions -3..-1)

    def convert(self, raw: bytes) -> bytes:
        arr = _decode(raw, self.sample_format, self.channels)
        if not self._resample or arr.size == 0:
            return _encode(arr)
        y, self._zi = lfilter(self._taps, 1.0, arr.astype(np.float64), zi=self._zi)
        n = y.size
        # Cubic (Catmull-Rom) interpolation needs one sample before and two
        # after each position, so positions up to n-2 are served now and the
        # rest on the next buffer, using the carried history.
        buf = np.concatenate((self._hist, y))          # buf index = position + 3
        limit = n - 2
        count = 0 if self._pos >= limit else int(np.ceil((limit - self._pos) / self._step))
        positions = self._pos + self._step * np.arange(count) + 3.0
        i = np.floor(positions).astype(np.int64)
        x = positions - i
        p0, p1, p2, p3 = buf[i - 1], buf[i], buf[i + 1], buf[i + 2]
        out = p1 + 0.5 * x * (p2 - p0 + x * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                                              + x * (3.0 * (p1 - p2) + p3 - p0)))
        self._pos = self._pos + self._step * count - n
        self._hist = buf[-3:].copy()
        return _encode(out.astype(np.float32))
