import typing as tp

import numpy as np
import numpy.typing as npt
from scipy.signal import butter, decimate, sosfilt

from egmlib.infra import typing as eltp

_AT = tp.TypeVar("_AT", bound=npt.NDArray)


def downsample(
    signal: _AT,
    frequency: eltp.Hz,
    *,
    target_frequency: eltp.Hz = 5000,
    ftype: tp.Literal["fir", "iir"] = "fir",
) -> _AT:
    q, r = divmod(frequency, target_frequency)
    if r != 0:
        raise ValueError(
            f"Текущая частота ({frequency}) должна делиться нацело на целевую ({target_frequency})"
        )

    if q == 1:
        return signal

    downsampled: _AT = decimate(signal, q=q, ftype=ftype, axis=-1)  # type: ignore

    return downsampled


def highpass_filter(
    signal: _AT, frequency: eltp.Hz, *, order: int = 2, critical_frequency: eltp.Hz = 250
) -> _AT:
    sos = butter(order, critical_frequency, "highpass", fs=frequency, output="sos")

    filtered: _AT = sosfilt(sos, signal, axis=-1)  # type: ignore

    return filtered


def moving_avg_filter(signal: _AT, *, size: int = 3) -> _AT:
    filtered: _AT = np.convolve(signal, np.ones(size)/size, mode="valid")  # type: ignore

    return filtered
