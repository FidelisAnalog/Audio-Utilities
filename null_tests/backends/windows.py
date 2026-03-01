"""
Windows audio backend using ASIO (primary) or WASAPI Exclusive (fallback).

ASIO provides exclusive access and is preferred for pro audio interfaces.
WASAPI Exclusive works with any Windows audio device.
"""

import os
import numpy as np
from typing import Optional

from .base import AudioBackend, AudioDevice, StreamConfig


# Enable ASIO support before importing sounddevice
os.environ['SD_ENABLE_ASIO'] = '1'

import sounddevice as sd


class WindowsAudioBackend(AudioBackend):
    """
    Windows audio backend with ASIO/WASAPI Exclusive support.
    
    Attempts ASIO first (for pro interfaces), falls back to WASAPI Exclusive.
    """
    
    def __init__(self):
        self._output_device: Optional[AudioDevice] = None
        self._input_device: Optional[AudioDevice] = None
        self._output_config: Optional[StreamConfig] = None
        self._input_config: Optional[StreamConfig] = None
        self._sd_output_id: Optional[int] = None
        self._sd_input_id: Optional[int] = None
        self._use_asio: bool = False
        self._host_api_map: dict = {}  # device index -> host api type
        
        # Check what host APIs are available
        self._available_apis = {api['name']: api for api in sd.query_hostapis()}
        self._has_asio = 'ASIO' in self._available_apis
        
        if self._has_asio:
            print("ASIO support detected")
        else:
            print("ASIO not available, will use WASAPI Exclusive")
    
    def enumerate_devices(self) -> list[AudioDevice]:
        """Return list of available audio devices, preferring ASIO."""
        devices = []
        seen_names = set()
        
        # First pass: collect ASIO devices (preferred)
        if self._has_asio:
            asio_api = self._available_apis['ASIO']
            for dev_idx in asio_api['devices']:
                dev = sd.query_devices(dev_idx)
                name = dev['name']
                
                # Get supported sample rates by testing
                sample_rates = self._probe_sample_rates(dev_idx)
                
                device = AudioDevice(
                    id=dev_idx,
                    name=f"{name} (ASIO)",
                    sample_rates=sample_rates,
                    max_channels_in=dev['max_input_channels'],
                    max_channels_out=dev['max_output_channels'],
                )
                devices.append(device)
                self._host_api_map[dev_idx] = 'ASIO'
                seen_names.add(name)
        
        # Second pass: collect WASAPI devices (fallback)
        if 'Windows WASAPI' in self._available_apis:
            wasapi_api = self._available_apis['Windows WASAPI']
            for dev_idx in wasapi_api['devices']:
                dev = sd.query_devices(dev_idx)
                name = dev['name']
                
                # Skip if we already have an ASIO version
                # (ASIO device names often match WASAPI names)
                base_name = name.replace(' (WASAPI)', '').strip()
                if base_name in seen_names:
                    continue
                
                sample_rates = self._probe_sample_rates(dev_idx)
                
                device = AudioDevice(
                    id=dev_idx,
                    name=f"{name}",
                    sample_rates=sample_rates,
                    max_channels_in=dev['max_input_channels'],
                    max_channels_out=dev['max_output_channels'],
                )
                devices.append(device)
                self._host_api_map[dev_idx] = 'WASAPI'
        
        return devices
    
    def _probe_sample_rates(self, device_idx: int) -> list[float]:
        """Probe which sample rates a device supports."""
        standard_rates = [44100, 48000, 88200, 96000, 176400, 192000]
        supported = []
        
        dev = sd.query_devices(device_idx)
        
        for rate in standard_rates:
            try:
                # Try to check if rate is supported
                if dev['max_output_channels'] > 0:
                    sd.check_output_settings(
                        device=device_idx,
                        samplerate=rate,
                        channels=1,
                    )
                elif dev['max_input_channels'] > 0:
                    sd.check_input_settings(
                        device=device_idx,
                        samplerate=rate,
                        channels=1,
                    )
                supported.append(float(rate))
            except sd.PortAudioError:
                pass
        
        # If probing failed, return the default rate
        if not supported:
            supported = [dev['default_samplerate']]
        
        return supported
    
    def _get_host_api(self, device_idx: int) -> str:
        """Get the host API type for a device."""
        dev = sd.query_devices(device_idx)
        api = sd.query_hostapis(dev['hostapi'])
        return api['name']
    
    def configure_output(self, device: AudioDevice, config: StreamConfig) -> None:
        """Configure output device."""
        if config.sample_rate not in device.sample_rates:
            raise ValueError(f"Sample rate {config.sample_rate} not supported by {device.name}")
        
        if config.channels > device.max_channels_out:
            raise ValueError(f"Requested {config.channels} channels but device only has {device.max_channels_out}")
        
        host_api = self._get_host_api(device.id)
        use_shared = getattr(self, 'shared_mode', False)
        
        self._output_device = device
        self._output_config = config
        self._sd_output_id = device.id
        
        if 'ASIO' in host_api:
            self._use_asio = True
            print(f"Output configured: {device.name} @ {config.sample_rate} Hz/{config.bit_depth}bit (ASIO)")
        elif use_shared:
            print(f"Output configured: {device.name} @ {config.sample_rate} Hz/{config.bit_depth}bit (WASAPI Shared)")
        else:
            print(f"Output configured: {device.name} @ {config.sample_rate} Hz/{config.bit_depth}bit (WASAPI Exclusive)")
    
    def configure_input(self, device: AudioDevice, config: StreamConfig) -> None:
        """Configure input device."""
        if config.sample_rate not in device.sample_rates:
            raise ValueError(f"Sample rate {config.sample_rate} not supported by {device.name}")
        
        if config.channels > device.max_channels_in:
            raise ValueError(f"Requested {config.channels} channels but device only has {device.max_channels_in}")
        
        host_api = self._get_host_api(device.id)
        use_shared = getattr(self, 'shared_mode', False)
        
        self._input_device = device
        self._input_config = config
        self._sd_input_id = device.id
        
        if 'ASIO' in host_api:
            print(f"Input configured: {device.name} @ {config.sample_rate} Hz/{config.bit_depth}bit (ASIO)")
        elif use_shared:
            print(f"Input configured: {device.name} @ {config.sample_rate} Hz/{config.bit_depth}bit (WASAPI Shared)")
        else:
            print(f"Input configured: {device.name} @ {config.sample_rate} Hz/{config.bit_depth}bit (WASAPI Exclusive)")
    
    def playrec(self, output_data: np.ndarray, config: StreamConfig) -> np.ndarray:
        """Simultaneously play and record."""
        if self._sd_output_id is None or self._sd_input_id is None:
            raise RuntimeError("Devices not configured. Call configure_output and configure_input first.")
        
        # Determine dtype based on bit depth
        if config.bit_depth == 16:
            dtype = 'int16'
        elif config.bit_depth == 24:
            dtype = 'int32'  # 24-bit packed into int32
        elif config.bit_depth == 32:
            dtype = 'int32'
        else:
            dtype = 'int32'
        
        # Ensure correct shape
        if output_data.ndim == 1:
            output_data = output_data.reshape(-1, 1)
        
        # Check for shared mode (set by gen_test.py for virtual devices)
        use_shared = getattr(self, 'shared_mode', False)
        
        # Build extra settings for exclusive access
        if self._use_asio:
            # ASIO is inherently exclusive
            extra_settings = None
        elif use_shared:
            # Shared mode for virtual devices
            extra_settings = None
        else:
            # WASAPI needs explicit exclusive mode
            wasapi_exclusive = sd.WasapiSettings(exclusive=True)
            extra_settings = wasapi_exclusive
        
        # Use playrec for simultaneous I/O
        try:
            recorded = sd.playrec(
                output_data,
                samplerate=int(config.sample_rate),
                channels=config.channels,
                input_mapping=list(range(1, config.channels + 1)),
                output_mapping=list(range(1, config.channels + 1)),
                device=(self._sd_input_id, self._sd_output_id),
                dtype=dtype,
                extra_settings=extra_settings,
            )
            sd.wait()
        except sd.PortAudioError as e:
            # If WASAPI exclusive fails, provide helpful message
            if 'exclusive' in str(e).lower() or 'wasapi' in str(e).lower():
                raise RuntimeError(
                    f"WASAPI Exclusive mode failed. Another application may be using the device. "
                    f"Error: {e}"
                )
            raise
        
        return recorded
    
    def release(self) -> None:
        """Release device access."""
        # sounddevice handles cleanup automatically
        # Just clear our state
        self._output_device = None
        self._input_device = None
        self._output_config = None
        self._input_config = None
        self._sd_output_id = None
        self._sd_input_id = None
        print("Released device access")
