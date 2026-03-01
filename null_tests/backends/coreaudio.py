"""
macOS CoreAudio backend using PyObjC for device control
and sounddevice for I/O.
"""

import ctypes
from ctypes import c_uint32, c_void_p, c_int32, c_double, byref, sizeof
from typing import Optional
import numpy as np

from .base import AudioBackend, AudioDevice, StreamConfig

# CoreAudio constants
kAudioObjectSystemObject = 1
kAudioHardwarePropertyDevices = 0x64657623  # 'dev#'
kAudioObjectPropertyScopeGlobal = 0x676C6F62  # 'glob'
kAudioObjectPropertyElementMain = 0
kAudioDevicePropertyNominalSampleRate = 0x6E737274  # 'nsrt'
kAudioDevicePropertyAvailableNominalSampleRates = 0x6E737223  # 'nsr#'
kAudioDevicePropertyHogMode = 0x6F696E6B  # 'oink'
kAudioDevicePropertyDeviceNameCFString = 0x6C6E616D  # 'lnam'
kAudioDevicePropertyStreamConfiguration = 0x736C6179  # 'slay'
kAudioObjectPropertyScopeInput = 0x696E7074  # 'inpt'
kAudioObjectPropertyScopeOutput = 0x6F757470  # 'outp'
kAudioDevicePropertyStreams = 0x73746D23  # 'stm#'
kAudioStreamPropertyPhysicalFormat = 0x70667420  # 'pft '
kAudioStreamPropertyAvailablePhysicalFormats = 0x70667423  # 'pft#'
kAudioFormatLinearPCM = 0x6C70636D  # 'lpcm'

# Format flags
kAudioFormatFlagIsFloat = 0x1
kAudioFormatFlagIsBigEndian = 0x2
kAudioFormatFlagIsSignedInteger = 0x4
kAudioFormatFlagIsPacked = 0x8
kAudioFormatFlagIsNonInterleaved = 0x20


class AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ('mSelector', c_uint32),
        ('mScope', c_uint32),
        ('mElement', c_uint32),
    ]


class AudioValueRange(ctypes.Structure):
    _fields_ = [
        ('mMinimum', c_double),
        ('mMaximum', c_double),
    ]


class AudioStreamBasicDescription(ctypes.Structure):
    _fields_ = [
        ('mSampleRate', c_double),
        ('mFormatID', c_uint32),
        ('mFormatFlags', c_uint32),
        ('mBytesPerPacket', c_uint32),
        ('mFramesPerPacket', c_uint32),
        ('mBytesPerFrame', c_uint32),
        ('mChannelsPerFrame', c_uint32),
        ('mBitsPerChannel', c_uint32),
        ('mReserved', c_uint32),
    ]


class AudioBufferList(ctypes.Structure):
    _fields_ = [
        ('mNumberBuffers', c_uint32),
        # Variable length array follows
    ]


# Load CoreAudio framework
_coreaudio = ctypes.CDLL('/System/Library/Frameworks/CoreAudio.framework/CoreAudio')

# Function signatures
_coreaudio.AudioObjectGetPropertyDataSize.argtypes = [
    c_uint32, ctypes.POINTER(AudioObjectPropertyAddress), c_uint32, c_void_p, ctypes.POINTER(c_uint32)
]
_coreaudio.AudioObjectGetPropertyDataSize.restype = c_int32

_coreaudio.AudioObjectGetPropertyData.argtypes = [
    c_uint32, ctypes.POINTER(AudioObjectPropertyAddress), c_uint32, c_void_p, ctypes.POINTER(c_uint32), c_void_p
]
_coreaudio.AudioObjectGetPropertyData.restype = c_int32

_coreaudio.AudioObjectSetPropertyData.argtypes = [
    c_uint32, ctypes.POINTER(AudioObjectPropertyAddress), c_uint32, c_void_p, c_uint32, c_void_p
]
_coreaudio.AudioObjectSetPropertyData.restype = c_int32


def _check_error(status: int, operation: str):
    """Raise exception if CoreAudio returned an error."""
    if status != 0:
        raise RuntimeError(f"CoreAudio error {status} during {operation}")


def _get_property_data_size(object_id: int, address: AudioObjectPropertyAddress) -> int:
    """Get size of property data in bytes."""
    size = c_uint32(0)
    status = _coreaudio.AudioObjectGetPropertyDataSize(
        object_id, byref(address), 0, None, byref(size)
    )
    _check_error(status, "get property data size")
    return size.value


def _get_device_name(device_id: int) -> str:
    """Get human-readable device name."""
    address = AudioObjectPropertyAddress(
        mSelector=kAudioDevicePropertyDeviceNameCFString,
        mScope=kAudioObjectPropertyScopeGlobal,
        mElement=kAudioObjectPropertyElementMain,
    )
    
    # Get CFStringRef
    cf_string = c_void_p()
    size = c_uint32(sizeof(c_void_p))
    status = _coreaudio.AudioObjectGetPropertyData(
        device_id, byref(address), 0, None, byref(size), byref(cf_string)
    )
    _check_error(status, "get device name")
    
    # Convert CFString to Python string
    CoreFoundation = ctypes.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
    CoreFoundation.CFStringGetCStringPtr.argtypes = [c_void_p, c_uint32]
    CoreFoundation.CFStringGetCStringPtr.restype = ctypes.c_char_p
    CoreFoundation.CFRelease.argtypes = [c_void_p]
    
    # Try fast path first
    name_ptr = CoreFoundation.CFStringGetCStringPtr(cf_string, 0x08000100)  # kCFStringEncodingUTF8
    if name_ptr:
        name = name_ptr.decode('utf-8')
    else:
        # Fallback: use CFStringGetCString
        CoreFoundation.CFStringGetCString.argtypes = [c_void_p, ctypes.c_char_p, c_int32, c_uint32]
        CoreFoundation.CFStringGetCString.restype = ctypes.c_bool
        buffer = ctypes.create_string_buffer(256)
        CoreFoundation.CFStringGetCString(cf_string, buffer, 256, 0x08000100)
        name = buffer.value.decode('utf-8')
    
    CoreFoundation.CFRelease(cf_string)
    return name


class AudioBuffer(ctypes.Structure):
    _fields_ = [
        ('mNumberChannels', c_uint32),
        ('mDataByteSize', c_uint32),
        ('mData', c_void_p),
    ]


def _get_channel_count(device_id: int, scope: int) -> int:
    """Get number of channels for input or output scope."""
    address = AudioObjectPropertyAddress(
        mSelector=kAudioDevicePropertyStreamConfiguration,
        mScope=scope,
        mElement=kAudioObjectPropertyElementMain,
    )
    
    # First check if property exists for this scope
    size = c_uint32(0)
    status = _coreaudio.AudioObjectGetPropertyDataSize(
        device_id, byref(address), 0, None, byref(size)
    )
    
    if status != 0 or size.value == 0:
        return 0
    
    # Allocate buffer for AudioBufferList
    # Structure: mNumberBuffers (u32) + padding (u32 on 64-bit for alignment) + array of AudioBuffer
    buffer = (ctypes.c_uint8 * size.value)()
    status = _coreaudio.AudioObjectGetPropertyData(
        device_id, byref(address), 0, None, byref(size), buffer
    )
    
    if status != 0:
        return 0
    
    # Parse AudioBufferList
    # First 4 bytes: mNumberBuffers
    num_buffers = int.from_bytes(buffer[0:4], byteorder='little')
    
    if num_buffers == 0:
        return 0
    
    # On 64-bit systems, AudioBufferList has 4 bytes padding after mNumberBuffers
    # Then each AudioBuffer is 16 bytes: mNumberChannels(4) + mDataByteSize(4) + mData(8)
    total_channels = 0
    offset = 8  # Skip mNumberBuffers (4) + padding (4)
    
    for _ in range(num_buffers):
        if offset + 4 > size.value:
            break
        channels = int.from_bytes(buffer[offset:offset+4], byteorder='little')
        total_channels += channels
        offset += 16  # sizeof(AudioBuffer) on 64-bit
    
    return total_channels


def _get_available_sample_rates(device_id: int) -> list[float]:
    """Get list of supported sample rates."""
    address = AudioObjectPropertyAddress(
        mSelector=kAudioDevicePropertyAvailableNominalSampleRates,
        mScope=kAudioObjectPropertyScopeGlobal,
        mElement=kAudioObjectPropertyElementMain,
    )
    
    size = _get_property_data_size(device_id, address)
    num_ranges = size // sizeof(AudioValueRange)
    
    ranges = (AudioValueRange * num_ranges)()
    size_ref = c_uint32(size)
    status = _coreaudio.AudioObjectGetPropertyData(
        device_id, byref(address), 0, None, byref(size_ref), ranges
    )
    _check_error(status, "get sample rates")
    
    # Common sample rates to check within ranges
    common_rates = [44100, 48000, 88200, 96000, 176400, 192000, 352800, 384000]
    available = []
    
    for r in ranges:
        if r.mMinimum == r.mMaximum:
            available.append(r.mMinimum)
        else:
            # Range of rates - check which common rates fall within
            for rate in common_rates:
                if r.mMinimum <= rate <= r.mMaximum:
                    available.append(float(rate))
    
    return sorted(set(available))


def _get_device_streams(device_id: int, scope: int) -> list[int]:
    """Get stream IDs for a device's input or output."""
    address = AudioObjectPropertyAddress(
        mSelector=kAudioDevicePropertyStreams,
        mScope=scope,
        mElement=kAudioObjectPropertyElementMain,
    )
    
    size = c_uint32(0)
    status = _coreaudio.AudioObjectGetPropertyDataSize(
        device_id, byref(address), 0, None, byref(size)
    )
    
    if status != 0 or size.value == 0:
        return []
    
    num_streams = size.value // sizeof(c_uint32)
    stream_ids = (c_uint32 * num_streams)()
    status = _coreaudio.AudioObjectGetPropertyData(
        device_id, byref(address), 0, None, byref(size), stream_ids
    )
    
    if status != 0:
        return []
    
    return list(stream_ids)


def _set_stream_physical_format(stream_id: int, sample_rate: float, bit_depth: int, channels: int) -> None:
    """Set the physical format of a stream."""
    address = AudioObjectPropertyAddress(
        mSelector=kAudioStreamPropertyPhysicalFormat,
        mScope=kAudioObjectPropertyScopeGlobal,
        mElement=kAudioObjectPropertyElementMain,
    )
    
    # Build format flags based on bit depth
    if bit_depth == 32:
        # 32-bit float
        format_flags = kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked
        bytes_per_sample = 4
    else:
        # Integer formats (16, 24)
        format_flags = kAudioFormatFlagIsSignedInteger | kAudioFormatFlagIsPacked
        if bit_depth == 24:
            bytes_per_sample = 3
        else:
            bytes_per_sample = bit_depth // 8
    
    asbd = AudioStreamBasicDescription(
        mSampleRate=sample_rate,
        mFormatID=kAudioFormatLinearPCM,
        mFormatFlags=format_flags,
        mBytesPerPacket=bytes_per_sample * channels,
        mFramesPerPacket=1,
        mBytesPerFrame=bytes_per_sample * channels,
        mChannelsPerFrame=channels,
        mBitsPerChannel=bit_depth,
        mReserved=0,
    )
    
    status = _coreaudio.AudioObjectSetPropertyData(
        stream_id, byref(address), 0, None, sizeof(AudioStreamBasicDescription), byref(asbd)
    )
    _check_error(status, f"set stream format to {sample_rate}Hz/{bit_depth}bit/{channels}ch")


def _get_stream_physical_format(stream_id: int) -> AudioStreamBasicDescription:
    """Get the current physical format of a stream."""
    address = AudioObjectPropertyAddress(
        mSelector=kAudioStreamPropertyPhysicalFormat,
        mScope=kAudioObjectPropertyScopeGlobal,
        mElement=kAudioObjectPropertyElementMain,
    )
    
    asbd = AudioStreamBasicDescription()
    size = c_uint32(sizeof(AudioStreamBasicDescription))
    status = _coreaudio.AudioObjectGetPropertyData(
        stream_id, byref(address), 0, None, byref(size), byref(asbd)
    )
    _check_error(status, "get stream format")
    return asbd


class CoreAudioBackend(AudioBackend):
    """macOS CoreAudio backend with hog mode support."""
    
    def __init__(self):
        self._output_device: Optional[AudioDevice] = None
        self._input_device: Optional[AudioDevice] = None
        self._output_config: Optional[StreamConfig] = None
        self._input_config: Optional[StreamConfig] = None
        self._hogged_devices: list[int] = []
        self._sd_output_id: Optional[int] = None
        self._sd_input_id: Optional[int] = None
    
    def enumerate_devices(self) -> list[AudioDevice]:
        """Enumerate all CoreAudio devices."""
        address = AudioObjectPropertyAddress(
            mSelector=kAudioHardwarePropertyDevices,
            mScope=kAudioObjectPropertyScopeGlobal,
            mElement=kAudioObjectPropertyElementMain,
        )
        
        size = _get_property_data_size(kAudioObjectSystemObject, address)
        num_devices = size // sizeof(c_uint32)
        
        device_ids = (c_uint32 * num_devices)()
        size_ref = c_uint32(size)
        status = _coreaudio.AudioObjectGetPropertyData(
            kAudioObjectSystemObject, byref(address), 0, None, byref(size_ref), device_ids
        )
        _check_error(status, "enumerate devices")
        
        devices = []
        for device_id in device_ids:
            try:
                name = _get_device_name(device_id)
                sample_rates = _get_available_sample_rates(device_id)
                channels_in = _get_channel_count(device_id, kAudioObjectPropertyScopeInput)
                channels_out = _get_channel_count(device_id, kAudioObjectPropertyScopeOutput)
                
                # Skip devices with no I/O
                if channels_in == 0 and channels_out == 0:
                    continue
                
                devices.append(AudioDevice(
                    id=device_id,
                    name=name,
                    sample_rates=sample_rates,
                    max_channels_in=channels_in,
                    max_channels_out=channels_out,
                ))
            except RuntimeError:
                continue
        
        return devices
    
    def _set_sample_rate(self, device_id: int, sample_rate: float) -> None:
        """Set device sample rate."""
        address = AudioObjectPropertyAddress(
            mSelector=kAudioDevicePropertyNominalSampleRate,
            mScope=kAudioObjectPropertyScopeGlobal,
            mElement=kAudioObjectPropertyElementMain,
        )
        
        rate = c_double(sample_rate)
        status = _coreaudio.AudioObjectSetPropertyData(
            device_id, byref(address), 0, None, sizeof(c_double), byref(rate)
        )
        _check_error(status, f"set sample rate to {sample_rate}")
    
    def _get_sample_rate(self, device_id: int) -> float:
        """Get current device sample rate."""
        address = AudioObjectPropertyAddress(
            mSelector=kAudioDevicePropertyNominalSampleRate,
            mScope=kAudioObjectPropertyScopeGlobal,
            mElement=kAudioObjectPropertyElementMain,
        )
        
        rate = c_double()
        size = c_uint32(sizeof(c_double))
        status = _coreaudio.AudioObjectGetPropertyData(
            device_id, byref(address), 0, None, byref(size), byref(rate)
        )
        _check_error(status, "get sample rate")
        return rate.value
    
    def _enable_hog_mode(self, device_id: int) -> None:
        """Take exclusive access to device."""
        address = AudioObjectPropertyAddress(
            mSelector=kAudioDevicePropertyHogMode,
            mScope=kAudioObjectPropertyScopeGlobal,
            mElement=kAudioObjectPropertyElementMain,
        )
        
        # Set hog mode to our process ID
        import os
        pid = c_int32(os.getpid())
        status = _coreaudio.AudioObjectSetPropertyData(
            device_id, byref(address), 0, None, sizeof(c_int32), byref(pid)
        )
        _check_error(status, "enable hog mode")
        self._hogged_devices.append(device_id)
    
    def _disable_hog_mode(self, device_id: int) -> None:
        """Release exclusive access to device."""
        address = AudioObjectPropertyAddress(
            mSelector=kAudioDevicePropertyHogMode,
            mScope=kAudioObjectPropertyScopeGlobal,
            mElement=kAudioObjectPropertyElementMain,
        )
        
        # Set hog mode to -1 to release
        pid = c_int32(-1)
        status = _coreaudio.AudioObjectSetPropertyData(
            device_id, byref(address), 0, None, sizeof(c_int32), byref(pid)
        )
        # Don't raise on error during cleanup
        if device_id in self._hogged_devices:
            self._hogged_devices.remove(device_id)
    
    def _find_sounddevice_id(self, device: AudioDevice, is_input: bool) -> int:
        """Find the sounddevice index matching a CoreAudio device."""
        import sounddevice as sd
        
        for idx, dev in enumerate(sd.query_devices()):
            # Match by name - sounddevice uses the same names as CoreAudio
            if device.name in dev['name']:
                if is_input and dev['max_input_channels'] > 0:
                    return idx
                elif not is_input and dev['max_output_channels'] > 0:
                    return idx
        
        raise ValueError(f"Could not find sounddevice index for {device.name}")
    
    def configure_output(self, device: AudioDevice, config: StreamConfig) -> None:
        """Configure output device with exclusive access."""
        if config.sample_rate not in device.sample_rates:
            raise ValueError(f"Sample rate {config.sample_rate} not supported by {device.name}")
        
        if config.channels > device.max_channels_out:
            raise ValueError(f"Requested {config.channels} channels but device only has {device.max_channels_out}")
        
        # Enable hog mode first
        self._enable_hog_mode(device.id)
        
        # Set sample rate at device level
        self._set_sample_rate(device.id, config.sample_rate)
        
        # Set stream physical format (sample rate + bit depth)
        import time
        streams = _get_device_streams(device.id, kAudioObjectPropertyScopeOutput)
        if streams:
            for stream_id in streams:
                try:
                    _set_stream_physical_format(stream_id, config.sample_rate, config.bit_depth, config.channels)
                except RuntimeError as e:
                    print(f"  Warning: Could not set format on stream {stream_id}: {e}")
        
        # Verify with retry - device may need time to switch
        max_attempts = 10
        for attempt in range(max_attempts):
            time.sleep(0.1)  # 100ms between checks
            actual_rate = self._get_sample_rate(device.id)
            if abs(actual_rate - config.sample_rate) <= 1:
                break
            if attempt < max_attempts - 1:
                # Retry setting the rate
                self._set_sample_rate(device.id, config.sample_rate)
        else:
            raise RuntimeError(f"Failed to set sample rate: requested {config.sample_rate}, got {actual_rate}")
        
        # Verify bit depth if possible
        actual_bits = None
        if streams:
            try:
                fmt = _get_stream_physical_format(streams[0])
                actual_bits = fmt.mBitsPerChannel
            except RuntimeError:
                pass
        
        self._output_device = device
        self._output_config = config
        self._sd_output_id = self._find_sounddevice_id(device, is_input=False)
        
        bits_str = f"/{actual_bits}bit" if actual_bits else ""
        print(f"Output configured: {device.name} @ {actual_rate} Hz{bits_str}, hog mode enabled")
    
    def configure_input(self, device: AudioDevice, config: StreamConfig) -> None:
        """Configure input device with exclusive access."""
        if config.sample_rate not in device.sample_rates:
            raise ValueError(f"Sample rate {config.sample_rate} not supported by {device.name}")
        
        if config.channels > device.max_channels_in:
            raise ValueError(f"Requested {config.channels} channels but device only has {device.max_channels_in}")
        
        # Enable hog mode first
        self._enable_hog_mode(device.id)
        
        # Set sample rate at device level
        self._set_sample_rate(device.id, config.sample_rate)
        
        # Set stream physical format (sample rate + bit depth)
        import time
        streams = _get_device_streams(device.id, kAudioObjectPropertyScopeInput)
        if streams:
            for stream_id in streams:
                try:
                    _set_stream_physical_format(stream_id, config.sample_rate, config.bit_depth, config.channels)
                except RuntimeError as e:
                    print(f"  Warning: Could not set format on stream {stream_id}: {e}")
        
        # Verify with retry - device may need time to switch
        max_attempts = 10
        for attempt in range(max_attempts):
            time.sleep(0.1)  # 100ms between checks
            actual_rate = self._get_sample_rate(device.id)
            if abs(actual_rate - config.sample_rate) <= 1:
                break
            if attempt < max_attempts - 1:
                # Retry setting the rate
                self._set_sample_rate(device.id, config.sample_rate)
        else:
            raise RuntimeError(f"Failed to set sample rate: requested {config.sample_rate}, got {actual_rate}")
        
        # Verify bit depth if possible
        actual_bits = None
        if streams:
            try:
                fmt = _get_stream_physical_format(streams[0])
                actual_bits = fmt.mBitsPerChannel
            except RuntimeError:
                pass
        
        self._input_device = device
        self._input_config = config
        self._sd_input_id = self._find_sounddevice_id(device, is_input=True)
        
        bits_str = f"/{actual_bits}bit" if actual_bits else ""
        print(f"Input configured: {device.name} @ {actual_rate} Hz{bits_str}, hog mode enabled")
    
    def playrec(self, output_data: np.ndarray, config: StreamConfig) -> np.ndarray:
        """Simultaneously play and record."""
        import sounddevice as sd
        
        if self._sd_output_id is None or self._sd_input_id is None:
            raise RuntimeError("Devices not configured. Call configure_output and configure_input first.")
        
        # Determine dtype based on bit depth
        # Note: numpy doesn't support int24, so we use int32 for both 24 and 32 bit
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
        
        # Use playrec for simultaneous I/O
        recorded = sd.playrec(
            output_data,
            samplerate=int(config.sample_rate),
            channels=config.channels,
            input_mapping=list(range(1, config.channels + 1)),
            output_mapping=list(range(1, config.channels + 1)),
            device=(self._sd_input_id, self._sd_output_id),
            dtype=dtype,
        )
        sd.wait()
        
        return recorded
    
    def release(self) -> None:
        """Release hog mode on all devices."""
        for device_id in list(self._hogged_devices):
            self._disable_hog_mode(device_id)
        print("Released exclusive access to all devices")
