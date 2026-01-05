"""
Platform-specific audio backends for bit-perfect I/O.
"""

import sys
from .base import AudioBackend, AudioDevice, StreamConfig


def get_backend() -> AudioBackend:
    """Return the appropriate backend for the current platform."""
    if sys.platform == 'darwin':
        from .coreaudio import CoreAudioBackend
        return CoreAudioBackend()
    elif sys.platform == 'win32':
        from .wasapi import WindowsAudioBackend
        return WindowsAudioBackend()
    else:
        raise NotImplementedError(f"No backend available for {sys.platform}")


__all__ = ['AudioBackend', 'AudioDevice', 'StreamConfig', 'get_backend']
