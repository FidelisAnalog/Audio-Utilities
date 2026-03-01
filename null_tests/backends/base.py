"""
Abstract base class for platform-specific audio backends.
Each backend must provide exclusive device access and bit-perfect I/O.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AudioDevice:
    """Represents an audio device."""
    id: any  # Platform-specific identifier
    name: str
    sample_rates: list[float]
    max_channels_in: int
    max_channels_out: int
    
    def __str__(self):
        direction = []
        if self.max_channels_out > 0:
            direction.append(f"{self.max_channels_out}ch out")
        if self.max_channels_in > 0:
            direction.append(f"{self.max_channels_in}ch in")
        return f"{self.name} ({', '.join(direction)})"


@dataclass 
class StreamConfig:
    """Configuration for an audio stream."""
    sample_rate: float
    bit_depth: int  # 16, 24, or 32
    channels: int


class AudioBackend(ABC):
    """
    Abstract base for platform-specific audio I/O.
    
    Implementations must provide:
    - Device enumeration
    - Exclusive access (hog mode / ASIO)
    - Sample rate configuration
    - Simultaneous play/record
    """
    
    @abstractmethod
    def enumerate_devices(self) -> list[AudioDevice]:
        """Return list of available audio devices."""
        pass
    
    @abstractmethod
    def configure_output(self, device: AudioDevice, config: StreamConfig) -> None:
        """
        Configure output device for exclusive access at specified format.
        Raises ValueError if format not supported.
        """
        pass
    
    @abstractmethod
    def configure_input(self, device: AudioDevice, config: StreamConfig) -> None:
        """
        Configure input device for exclusive access at specified format.
        Raises ValueError if format not supported.
        """
        pass
    
    @abstractmethod
    def playrec(self, output_data: np.ndarray, config: StreamConfig) -> np.ndarray:
        """
        Simultaneously play output_data and record input.
        
        Args:
            output_data: Audio to play, shape (samples, channels)
            config: Stream configuration (must match configured devices)
            
        Returns:
            Recorded audio, shape (samples, channels)
        """
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release exclusive access to all devices."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False
