#!/usr/bin/env python3
"""
EEG Signal Processing for Visuospatial Attention Detection
Processes O1, O2, Fz, Cz channels for attention lateralization
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, welch
import threading
from collections import deque

class EEGProcessor:
    def __init__(self, sample_rate=250):
        """
        Initialize EEG processor for attention detection
        
        Args:
            sample_rate: EEG sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.channels = ['O1', 'O2', 'Fz', 'Cz']
        self.n_channels = len(self.channels)
        
        # Frequency bands of interest
        self.alpha_band = [8, 13]    # Alpha band (8-13 Hz)
        self.beta_band = [13, 30]    # Beta band (13-30 Hz)
        self.theta_band = [4, 8]     # Theta band (4-8 Hz)
        
        # Filter parameters
        self.lowpass_freq = 50       # Low-pass filter cutoff
        self.highpass_freq = 1       # High-pass filter cutoff
        self.notch_freq = 50         # Notch filter for power line interference
        
        # Window parameters for spectral analysis
        self.window_size = int(2 * sample_rate)  # 2 seconds
        self.overlap = int(0.5 * sample_rate)    # 0.5 seconds overlap
        
        # Buffer for continuous processing
        self.processing_buffer = deque(maxlen=self.window_size)
        
        # Pre-compute filters
        self._setup_filters()
        
        # Feature history for smoothing
        self.feature_history = {
            'alpha_o1': deque(maxlen=10),
            'alpha_o2': deque(maxlen=10),
            'alpha_fz': deque(maxlen=10),
            'alpha_cz': deque(maxlen=10),
            'asymmetry': deque(maxlen=10)
        }
        
        print(f"EEG Processor initialized for {self.n_channels} channels at {sample_rate} Hz")
    
    def _setup_filters(self):
        """Setup digital filters for preprocessing"""
        # Bandpass filter (1-50 Hz)
        self.bp_b, self.bp_a = butter(4, [self.highpass_freq, self.lowpass_freq], 
                                     btype='band', fs=self.sample_rate)
        
        # Notch filter (50 Hz)
        self.notch_b, self.notch_a = butter(4, [self.notch_freq - 2, self.notch_freq + 2], 
                                           btype='bandstop', fs=self.sample_rate)
        
        print("Digital filters initialized")
    
    def preprocess_data(self, data):
        """
        Preprocess EEG data with filtering
        
        Args:
            data: EEG data array [n_samples, n_channels]
            
        Returns:
            np.ndarray: Preprocessed data
        """
        if data is None or len(data) == 0:
            return None
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        processed_data = np.zeros_like(data)
        
        # Apply filters to each channel
        for ch in range(data.shape[1]):
            channel_data = data[:, ch]
            
            # Apply bandpass filter
            filtered = filtfilt(self.bp_b, self.bp_a, channel_data)
            
            # Apply notch filter
            filtered = filtfilt(self.notch_b, self.notch_a, filtered)
            
            processed_data[:, ch] = filtered
        
        return processed_data
    
    def compute_power_spectral_density(self, data, channel_idx=0):
        """
        Compute power spectral density for a channel
        
        Args:
            data: EEG data for single channel
            channel_idx: Channel index (for labeling)
            
        Returns:
            tuple: (frequencies, power spectral density)
        """
        if len(data) < self.window_size // 2:
            return None, None
        
        # Compute PSD using Welch's method
        frequencies, psd = welch(data, fs=self.sample_rate, 
                                nperseg=min(len(data), self.window_size // 2),
                                noverlap=self.overlap // 2,
                                window='hann')
        
        return frequencies, psd
    
    def extract_band_power(self, frequencies, psd, band):
        """
        Extract power in a specific frequency band
        
        Args:
            frequencies: Frequency array from PSD
            psd: Power spectral density
            band: Frequency band [low, high] in Hz
            
        Returns:
            float: Band power
        """
        if frequencies is None or psd is None:
            return 0.0
        
        # Find indices for frequency band
        band_idx = np.logical_and(frequencies >= band[0], frequencies <= band[1])
        
        # Calculate band power (sum of PSD in band)
        band_power = np.sum(psd[band_idx])
        
        return band_power
    
    def compute_alpha_asymmetry(self, alpha_left, alpha_right):
        """
        Compute alpha asymmetry index
        
        Args:
            alpha_left: Alpha power in left hemisphere (O1)
            alpha_right: Alpha power in right hemisphere (O2)
            
        Returns:
            float: Asymmetry index (positive = right attention, negative = left attention)
        """
        # Alpha asymmetry: log(right) - log(left)
        # Higher alpha = less activation, so we flip the sign
        asymmetry = np.log(alpha_left) - np.log(alpha_right)
        
        
        return asymmetry
    
    def smooth_features(self, feature_dict):
        """
        Apply temporal smoothing to features
        
        Args:
            feature_dict: Dictionary of current features
            
        Returns:
            dict: Smoothed features
        """
        smoothed = {}
        
        for key, value in feature_dict.items():
            if key in self.feature_history:
                self.feature_history[key].append(value)
                # Use median for robust smoothing
                smoothed[key] = np.median(list(self.feature_history[key]))
            else:
                smoothed[key] = value
        
        return smoothed
    
    def process_realtime(self, data):
        """
        Process real-time EEG data for attention detection
        
        Args:
            data: Raw EEG data [n_samples, 4] for O1, O2, Fz, Cz
            
        Returns:
            tuple: (alpha_o1, alpha_o2, alpha_fz, alpha_cz, asymmetry)
        """
        if data is None or len(data) == 0:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        
        try:
            # Preprocess data
            processed_data = self.preprocess_data(data)
            # processed_data = data                         # Use the raw, clean data directly
            if processed_data is None:
                return 0.0, 0.0, 0.0, 0.0, 0.0
            
            # Extract features for each channel
            alpha_powers = []
            
            for ch in range(min(4, processed_data.shape[1])):
                # Get channel data
                channel_data = processed_data[:, ch]
                
                # Compute PSD
                frequencies, psd = self.compute_power_spectral_density(channel_data, ch)
                
                # Extract alpha band power
                alpha_power = self.extract_band_power(frequencies, psd, self.alpha_band)
                alpha_powers.append(alpha_power)
            
            # Ensure we have 4 channel values
            while len(alpha_powers) < 2:
                alpha_powers.append(0.0)
            
            alpha_o1, alpha_o2, alpha_fz, alpha_cz = alpha_powers[:4]
            
            # Compute asymmetry (O1 vs O2)
            asymmetry = self.compute_alpha_asymmetry(alpha_o1, alpha_o2)
            
            # Create feature dictionary
            features = {
                'alpha_o1': alpha_o1,
                'alpha_o2': alpha_o2,
                'alpha_fz': alpha_fz,
                'alpha_cz': alpha_cz,
                'asymmetry': asymmetry
            }
            
            # Apply smoothing
            smoothed_features = self.smooth_features(features)
            
            return (smoothed_features['alpha_o1'], 
                   smoothed_features['alpha_o2'],
                   smoothed_features['alpha_fz'],
                   smoothed_features['alpha_cz'],
                   smoothed_features['asymmetry'])
        
        except Exception as e:
            print(f"Error in real-time processing: {e}")
            return 0.0, 0.0, 0.0, 0.0, 0.0
    
    def process_batch(self, data):
        """
        Process batch EEG data for offline analysis
        
        Args:
            data: EEG data array [n_samples, n_channels]
            
        Returns:
            dict: Comprehensive feature dictionary
        """
        if data is None or len(data) == 0:
            return {}
        
        # Preprocess data
        processed_data = data
        if processed_data is None:
            return {}
        
        features = {}
        
        # Process each channel
        for ch, channel_name in enumerate(self.channels):
            if ch >= processed_data.shape[1]:
                break
            
            channel_data = processed_data[:, ch]
            
            # Compute PSD
            frequencies, psd = self.compute_power_spectral_density(channel_data, ch)
            
            if frequencies is not None and psd is not None:
                # Extract power in different bands
                features[f'{channel_name}_alpha'] = self.extract_band_power(frequencies, psd, self.alpha_band)
                features[f'{channel_name}_beta'] = self.extract_band_power(frequencies, psd, self.beta_band)
                features[f'{channel_name}_theta'] = self.extract_band_power(frequencies, psd, self.theta_band)
                
                # Store PSD for later analysis
                features[f'{channel_name}_frequencies'] = frequencies
                features[f'{channel_name}_psd'] = psd
        
        # Compute asymmetry indices
        if 'O1_alpha' in features and 'O2_alpha' in features:
            features['alpha_asymmetry'] = self.compute_alpha_asymmetry(features['O1_alpha'], features['O2_alpha'])
        
        if 'Fz_alpha' in features and 'Cz_alpha' in features:
            features['frontal_central_ratio'] = features['Fz_alpha'] / (features['Cz_alpha'] + 1e-10)
        
        return features
    
    def detect_attention_direction(self, asymmetry, threshold=0.1):
        """
        Detect attention direction from asymmetry
        
        Args:
            asymmetry: Alpha asymmetry value
            threshold: Threshold for direction detection
            
        Returns:
            str: 'left', 'right', or 'center'
        """
        if asymmetry > threshold:
            return 'right'
        elif asymmetry < -threshold:
            return 'left'
        else:
            return 'center'
    
    def get_attention_confidence(self, asymmetry):
        """
        Get confidence score for attention direction
        
        Args:
            asymmetry: Alpha asymmetry value
            
        Returns:
            float: Confidence score (0-1)
        """
        # Confidence based on magnitude of asymmetry
        confidence = min(abs(asymmetry) / 2.0, 1.0)  # Normalize to 0-1
        return confidence
    
    def reset_history(self):
        """Reset feature history buffers"""
        for key in self.feature_history:
            self.feature_history[key].clear()
    
    def get_processor_info(self):
        """Get processor configuration info"""
        return {
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'alpha_band': self.alpha_band,
            'beta_band': self.beta_band,
            'theta_band': self.theta_band,
            'window_size': self.window_size,
            'overlap': self.overlap
        }

def test_eeg_processor():
    """Test function for EEG processor"""
    print("Testing EEG processor...")
    
    # Create processor
    processor = EEGProcessor(sample_rate=250)
    
    # Generate test data (2 seconds of 4-channel EEG)
    duration = 2.0
    sample_rate = 250
    n_samples = int(duration * sample_rate)
    
    # Simulate EEG data with some alpha activity
    t = np.linspace(0, duration, n_samples)
    
    # O1 channel (left occipital) - more alpha
    o1 = np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(n_samples)
    
    # O2 channel (right occipital) - less alpha
    o2 = 0.5 * np.sin(2 * np.pi * 10 * t) + 0.5 * np.random.randn(n_samples)
    
    # Fz channel (frontal)
    fz = 0.3 * np.sin(2 * np.pi * 6 * t) + 0.3 * np.random.randn(n_samples)
    
    # Cz channel (central)
    cz = 0.2 * np.sin(2 * np.pi * 8 * t) + 0.3 * np.random.randn(n_samples)
    
    # Combine channels
    test_data = np.column_stack([o1, o2, fz, cz])
    
    print(f"Test data shape: {test_data.shape}")
    
    # Test real-time processing
    print("\nTesting real-time processing...")
    alpha_o1, alpha_o2, alpha_fz, alpha_cz, asymmetry = processor.process_realtime(test_data)
    
    print(f"Alpha O1: {alpha_o1:.3f}")
    print(f"Alpha O2: {alpha_o2:.3f}")
    print(f"Alpha Fz: {alpha_fz:.3f}")
    print(f"Alpha Cz: {alpha_cz:.3f}")
    print(f"Asymmetry: {asymmetry:.3f}")
    
    # Test attention detection
    direction = processor.detect_attention_direction(asymmetry)
    confidence = processor.get_attention_confidence(asymmetry)
    
    print(f"Attention direction: {direction}")
    print(f"Confidence: {confidence:.3f}")
    
    # Test batch processing
    print("\nTesting batch processing...")
    features = processor.process_batch(test_data)
    
    print("Extracted features:")
    for key, value in features.items():
        if not key.endswith('_frequencies') and not key.endswith('_psd'):
            print(f"  {key}: {value:.3f}")
    
    print("Test completed successfully!")

if __name__ == "__main__":
    test_eeg_processor()