"""System audio through WASAPI loopback — no virtual cable.

Windows makes whatever plays on an output device available for recording. That
removes VB-Cable (a driver, administrator rights, a reboot) and the Zoom setup
from the installation: the respondent channel is taken straight from the
headphones Zoom is already playing into.

PortAudio, which sounddevice runs on, does not expose this — in the 19.7-devel
build WasapiSettings has no loopback parameter, and no "… [Loopback]" devices
appear in the enumeration. So on Windows the system-audio channel is held by a
separate library, PyAudioWPatch (a fork of PyAudio with loopback support). It
lives only here, behind this module's facade.

PyAudio and sounddevice number devices independently, so loopback devices are
exposed outwards with a LOOPBACK_INDEX_BASE offset: the rest of the code keeps
working with a single flat int device identifier.
"""
from __future__ import annotations

import logging
import sys

import numpy as np

log = logging.getLogger("ilh.audio.loopback")

LOOPBACK_INDEX_BASE = 10_000


def available() -> bool:
    """Whether this machine has the loopback route (Windows plus PyAudioWPatch installed)."""
    if sys.platform != "win32":
        return False
    try:
        import pyaudiowpatch  # noqa: F401
    except ImportError:
        return False
    return True


def is_loopback_index(index: int | None) -> bool:
    return index is not None and index >= LOOPBACK_INDEX_BASE


def list_loopback_devices() -> list[dict]:
    """Output devices available for recording. Same format as list_input_devices()."""
    if not available():
        return []
    import pyaudiowpatch as pa

    p = pa.PyAudio()
    try:
        try:
            wasapi = p.get_host_api_info_by_type(pa.paWASAPI)
            default_out = str(p.get_device_info_by_index(wasapi["defaultOutputDevice"])["name"])
        except (OSError, KeyError, ValueError):
            default_out = ""

        devices = []
        for dev in p.get_loopback_device_info_generator():
            raw_name = str(dev["name"])
            devices.append(
                {
                    "index": LOOPBACK_INDEX_BASE + int(dev["index"]),
                    "name": raw_name.removesuffix(" [Loopback]"),
                    "hostapi": "WASAPI loopback",
                    "max_input_channels": int(dev["maxInputChannels"]),
                    "default_samplerate": float(dev["defaultSampleRate"]),
                    "is_default_input": False,
                    "is_loopback": True,
                    # The device the system is playing into right now: that is what we offer.
                    "is_default_loopback": bool(default_out and default_out in raw_name),
                }
            )
        return devices
    except OSError as e:  # no WASAPI (Wine, exotic builds) — no reason to crash
        log.warning("Loopback devices are unavailable: %s", e)
        return []
    finally:
        p.terminate()


def device_info(index: int) -> dict:
    """Description of a loopback device by its external (offset) index."""
    for dev in list_loopback_devices():
        if dev["index"] == index:
            return dev
    raise RuntimeError(
        f"Loopback device #{index - LOOPBACK_INDEX_BASE} has disappeared — "
        "refresh the device list"
    )


class LoopbackStream:
    """Capture from an output device.

    The interface deliberately mirrors sounddevice.InputStream in the part that
    ChannelCapture uses (start/stop/close), so the channel need not know which of
    the two routes the audio arrived by.
    """

    def __init__(self, info: dict, on_block, blocksize: int = 1024):
        self.pa_index = int(info["index"]) - LOOPBACK_INDEX_BASE
        self.channels = max(1, int(info["max_input_channels"]))
        self.samplerate = int(info["default_samplerate"])
        self.blocksize = blocksize
        self._on_block = on_block
        self._pa = None
        self._stream = None

    def start(self) -> None:
        import pyaudiowpatch as pa

        self._pa = pa.PyAudio()
        try:
            self._stream = self._pa.open(
                format=pa.paFloat32,
                channels=self.channels,
                rate=self.samplerate,
                input=True,
                input_device_index=self.pa_index,
                frames_per_buffer=self.blocksize,
                stream_callback=self._callback,
            )
        except Exception:
            self._pa.terminate()
            self._pa = None
            raise
        self._stream.start_stream()

    def _callback(self, in_data, frame_count, time_info, status):
        import pyaudiowpatch as pa

        block = np.frombuffer(in_data, dtype=np.float32)
        if self.channels > 1:
            # Stereo to mono by averaging rather than by taking the first channel:
            # some applications pan the voice, and the left channel may be empty.
            block = block.reshape(-1, self.channels).mean(axis=1)
        self._on_block(np.ascontiguousarray(block, dtype=np.float32), bool(status))
        return (None, pa.paContinue)

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None
