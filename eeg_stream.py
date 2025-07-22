#!/usr/bin/env python3
"""
EEG Stream Handler for OpenBCI via Lab Streaming Layer (LSL)
Connects to OpenBCI GUI and streams O1, O2, P3, P4 channels at 250 Hz
"""
import numpy as np
import time
from collections import deque
import threading
from pylsl import StreamInlet, resolve_streams

class EEGStreamer:
    def __init__(self, buffer_size=1250):  # 5 seconds at 250 Hz
        self.buffer_size = buffer_size
        self.sample_rate = 250
        # MODIFIED: Updated channel names
        self.channels = ['O1', 'O2', 'P3', 'P4']
        self.n_channels = len(self.channels)
        
        self.inlet = None
        self.stream_info = None
        self.data_buffer = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        
        self.streaming = False
        self.stream_thread = None
        self.data_lock = threading.Lock()
        self.connected = False
        
        print(f"EEG Streamer initialized for channels: {self.channels}")
        print(f"Expected sample rate: {self.sample_rate} Hz")
    
    def find_eeg_stream(self, timeout=10):
        print("Searching for OpenBCI EEG stream via LSL...")
        streams = resolve_streams(timeout)
        streams = [s for s in streams if s.type() == 'EEG']
        
        if not streams:
            print("No EEG streams found!")
            return False
        
        self.stream_info = streams[0]
        print(f"Found EEG stream: {self.stream_info.name()}")
        print(f"Stream type: {self.stream_info.type()}")
        print(f"Channel count: {self.stream_info.channel_count()}")
        print(f"Sample rate: {self.stream_info.nominal_srate()} Hz")
        
        if self.stream_info.channel_count() < 4:
            # MODIFIED: Updated error message
            print("Error: Stream must have at least 4 channels (O1, O2, P3, P4)")
            return False
        return True
    
    def connect(self):
        try:
            if not self.find_eeg_stream():
                return False
            
            self.inlet = StreamInlet(self.stream_info)
            actual_channels = self.inlet.info().channel_count()
            actual_rate = self.inlet.info().nominal_srate()
            print(f"Connected to stream with {actual_channels} channels at {actual_rate} Hz")
            
            if actual_rate != self.sample_rate:
                print(f"Warning: Expected {self.sample_rate} Hz, got {actual_rate} Hz")
                self.sample_rate = int(actual_rate)
            
            self.connected = True
            return True
        except Exception as e:
            print(f"Failed to connect to EEG stream: {e}")
            return False
    
    def start_streaming(self):
        if not self.connected:
            print("Error: Not connected to EEG stream")
            return False
        if self.streaming:
            print("Already streaming")
            return True
        
        self.streaming = True
        self.stream_thread = threading.Thread(target=self._stream_worker)
        self.stream_thread.daemon = True
        self.stream_thread.start()
        print("EEG streaming started")
        return True
    
    def _stream_worker(self):
        while self.streaming:
            try:
                samples, timestamps = self.inlet.pull_chunk(timeout=0.1, max_samples=32)
                if samples:
                    with self.data_lock:
                        for i, sample in enumerate(samples):
                            if len(sample) >= 6: # Ensure we have enough channels for the max index
                                # MODIFIED: Mapping to new channels, using your specified indices
                                channel_data = [
                                    sample[0],  # O1
                                    sample[1],  # O2
                                    sample[2],  # P3
                                    sample[3]   # P4
                                ]
                                self.data_buffer.append(channel_data)
                                self.timestamps.append(timestamps[i])
            except Exception as e:
                if self.streaming:
                    print(f"Streaming error: {e}")
                time.sleep(0.001)
    
    def get_latest_data(self, n_samples=125):
        if not self.streaming:
            return None
        with self.data_lock:
            if len(self.data_buffer) < n_samples:
                return None
            recent_data = list(self.data_buffer)[-n_samples:]
            return np.array(recent_data)

    # ... (the rest of the functions: get_buffer_size, clear_buffer, etc. remain the same) ...
    
    def disconnect(self):
        self.stop_streaming()
        if self.inlet:
            # self.inlet.close() # Commented out for compatibility
            self.inlet = None
        self.clear_buffer()
        self.connected = False
        print("Disconnected from EEG stream")