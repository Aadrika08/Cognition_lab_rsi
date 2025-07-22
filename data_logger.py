#!/usr/bin/env python3
"""
Data Logger for BCI Visuospatial Attention Game
"""
import csv
import os
from datetime import datetime
import threading

class DataLogger:
    def __init__(self, participant_name):
        self.participant_name = participant_name
        self.session_start = datetime.now()
        timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
        self.filename = f"bci_attention_data_{participant_name}_{timestamp}.csv"
        
        self.data_dir = "data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        self.filepath = os.path.join(self.data_dir, self.filename)
        
        # MODIFIED: Updated CSV headers
        self.headers = [
            'participant_name', 'session_start', 'trial_number',
            'trial_timestamp', 'trial_time_elapsed', 'target_side',
            'alpha_o1', 'alpha_o2', 'alpha_p3', 'alpha_p4',
            'alpha_asymmetry', 'attention_direction', 'correct',
            'reaction_time', 'confidence', 'ball_position_x',
            'ball_target_x', 'trial_duration', 'notes'
        ]
        
        self.write_lock = threading.Lock()
        self._initialize_csv()
        print(f"Data logger initialized for participant: {participant_name}")
        print(f"Data will be saved to: {self.filepath}")
    
    def _initialize_csv(self):
        try:
            with open(self.filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.headers)
                writer.writeheader()
            print(f"CSV file created: {self.filepath}")
        except Exception as e:
            print(f"Error initializing CSV file: {e}")
            raise
    
    # MODIFIED: Updated function signature
    def log_trial(self, trial_number, timestamp, target_side, 
                  alpha_o1, alpha_o2, alpha_p3, alpha_p4, 
                  alpha_asymmetry, attention_direction, correct, 
                  reaction_time, confidence, ball_position_x=None, 
                  ball_target_x=None, notes=""):
        time_elapsed = (timestamp - self.session_start).total_seconds()
        
        # MODIFIED: Updated row data
        row_data = {
            'participant_name': self.participant_name,
            'session_start': self.session_start.strftime("%Y-%m-%d %H:%M:%S"),
            'trial_number': trial_number,
            'trial_timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            'trial_time_elapsed': round(time_elapsed, 3),
            'target_side': target_side,
            'alpha_o1': round(alpha_o1, 6),
            'alpha_o2': round(alpha_o2, 6),
            'alpha_p3': round(alpha_p3, 6),
            'alpha_p4': round(alpha_p4, 6),
            'alpha_asymmetry': round(alpha_asymmetry, 6),
            'attention_direction': attention_direction,
            'correct': correct,
            'reaction_time': round(reaction_time, 3),
            'confidence': round(confidence, 6),
            'ball_position_x': ball_position_x,
            'ball_target_x': ball_target_x,
            'trial_duration': 5.0,
            'notes': notes
        }
        
        with self.write_lock:
            try:
                with open(self.filepath, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.headers)
                    writer.writerow(row_data)
            except Exception as e:
                print(f"Error writing to CSV: {e}")
    
    # ... (the rest of the logger class remains the same) ...
    
    def log_session_info(self, info_dict):
        """
        Log session information to a separate file
        
        Args:
            info_dict: Dictionary containing session information
        """
        info_filename = f"session_info_{self.participant_name}_{self.session_start.strftime('%Y%m%d_%H%M%S')}.txt"
        info_filepath = os.path.join(self.data_dir, info_filename)
        
        try:
            with open(info_filepath, 'w', encoding='utf-8') as f:
                f.write("BCI Visuospatial Attention Game - Session Information\n")
                f.write("=" * 50 + "\n\n")
                
                for key, value in info_dict.items():
                    f.write(f"{key}: {value}\n")
                
                f.write(f"\nSession started: {self.session_start}\n")
                f.write(f"Data file: {self.filename}\n")
            
            print(f"Session info saved to: {info_filepath}")
            
        except Exception as e:
            print(f"Error saving session info: {e}")
    
    def log_eeg_quality(self, quality_metrics):
        """
        Log EEG signal quality metrics
        
        Args:
            quality_metrics: Dictionary with signal quality information
        """
        quality_filename = f"eeg_quality_{self.participant_name}_{self.session_start.strftime('%Y%m%d_%H%M%S')}.csv"
        quality_filepath = os.path.join(self.data_dir, quality_filename)
        
        quality_headers = ['timestamp', 'channel', 'signal_quality', 'impedance', 'artifacts', 'notes']
        
        # Initialize quality CSV if it doesn't exist
        if not os.path.exists(quality_filepath):
            with open(quality_filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=quality_headers)
                writer.writeheader()
        
        # Write quality data
        with self.write_lock:
            try:
                with open(quality_filepath, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=quality_headers)
                    
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    
                    for channel, metrics in quality_metrics.items():
                        row = {
                            'timestamp': timestamp,
                            'channel': channel,
                            'signal_quality': metrics.get('quality', 'unknown'),
                            'impedance': metrics.get('impedance', ''),
                            'artifacts': metrics.get('artifacts', ''),
                            'notes': metrics.get('notes', '')
                        }
                        writer.writerow(row)
                
            except Exception as e:
                print(f"Error writing EEG quality data: {e}")
    
    def get_session_stats(self):
        """
        Get basic session statistics
        
        Returns:
            dict: Session statistics
        """
        stats = {
            'participant': self.participant_name,
            'session_start': self.session_start,
            'data_file': self.filename,
            'file_size': 0,
            'trials_logged': 0
        }
        
        # Get file size
        try:
            if os.path.exists(self.filepath):
                stats['file_size'] = os.path.getsize(self.filepath)
                
                # Count trials by counting lines (minus header)
                with open(self.filepath, 'r', encoding='utf-8') as f:
                    stats['trials_logged'] = sum(1 for line in f) - 1
                    
        except Exception as e:
            print(f"Error getting session stats: {e}")
        
        return stats
    
    def backup_data(self):
        """Create a backup of the current data file"""
        if not os.path.exists(self.filepath):
            return False
        
        try:
            backup_filename = f"backup_{self.filename}"
            backup_filepath = os.path.join(self.data_dir, backup_filename)
            
            import shutil
            shutil.copy2(self.filepath, backup_filepath)
            
            print(f"Data backup created: {backup_filepath}")
            return True
            
        except Exception as e:
            print(f"Error creating backup: {e}")
            return False
    
    def validate_data_integrity(self):
        """
        Validate the integrity of logged data
        
        Returns:
            dict: Validation results
        """
        validation_results = {
            'file_exists': False,
            'readable': False,
            'header_valid': False,
            'row_count': 0,
            'errors': []
        }
        
        try:
            # Check if file exists
            validation_results['file_exists'] = os.path.exists(self.filepath)
            
            if validation_results['file_exists']:
                # Try to read the file
                with open(self.filepath, 'r', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)
                    validation_results['readable'] = True
                    
                    # Check headers
                    if reader.fieldnames == self.headers:
                        validation_results['header_valid'] = True
                    else:
                        validation_results['errors'].append("Header mismatch")
                    
                    # Count rows and check for basic data integrity
                    row_count = 0
                    for row in reader:
                        row_count += 1
                        
                        # Basic validation checks
                        if not row.get('trial_number', '').isdigit():
                            validation_results['errors'].append(f"Invalid trial number in row {row_count}")
                        
                        if row.get('correct') not in ['0', '1']:
                            validation_results['errors'].append(f"Invalid correct value in row {row_count}")
                    
                    validation_results['row_count'] = row_count
                    
        except Exception as e:
            validation_results['errors'].append(f"Validation error: {e}")
        
        return validation_results
    
    def close(self):
        """Close the logger and perform final operations"""
        try:
            # Create backup
            self.backup_data()
            
            # Validate data integrity
            validation = self.validate_data_integrity()
            
            # Get final stats
            stats = self.get_session_stats()
            
            print(f"\nSession completed for {self.participant_name}")
            print(f"Total trials logged: {stats['trials_logged']}")
            print(f"Data file size: {stats['file_size']} bytes")
            
            if validation['errors']:
                print("Data validation warnings:")
                for error in validation['errors']:
                    print(f"  - {error}")
            else:
                print("Data validation: PASSED")
            
            print(f"Data saved to: {self.filepath}")
            
        except Exception as e:
            print(f"Error closing logger: {e}")

def test_data_logger():
    """Test function for data logger"""
    print("Testing data logger...")
    
    # Create test logger
    logger = DataLogger("test_participant")
    
    # Log some test session info
    session_info = {
        'experiment_version': '1.0',
        'eeg_system': 'OpenBCI',
        'channels': 'O1, O2, Fz, Cz',
        'sample_rate': '250 Hz',
        'total_trials': 150
    }
    logger.log_session_info(session_info)
    
    # Log some test trials
    base_time = datetime.now()
    
    for trial in range(1, 6):  # Test 5 trials
        trial_time = base_time.replace(second=base_time.second + trial)
        
        logger.log_trial(
            trial_number=trial,
            timestamp=trial_time,
            target_side='left' if trial % 2 == 0 else 'right',
            alpha_o1=10.5 + trial * 0.1,
            alpha_o2=9.8 + trial * 0.15,
            alpha_fz=8.2 + trial * 0.05,
            alpha_cz=7.9 + trial * 0.08,
            alpha_asymmetry=0.1 * (trial % 3 - 1),
            attention_direction='left' if trial % 2 == 0 else 'right',
            correct=1 if trial % 3 != 0 else 0,
            reaction_time=0.5 + trial * 0.1,
            confidence=0.7 + trial * 0.05,
            ball_position_x=400 + trial * 10,
            ball_target_x=450 + trial * 5,
            notes=f"Test trial {trial}"
        )
    
    # Test EEG quality logging
    quality_metrics = {
        'O1': {'quality': 'good', 'impedance': '5kΩ', 'artifacts': 'none'},
        'O2': {'quality': 'good', 'impedance': '7kΩ', 'artifacts': 'none'},
        'Fz': {'quality': 'fair', 'impedance': '12kΩ', 'artifacts': 'blink'},
        'Cz': {'quality': 'good', 'impedance': '6kΩ', 'artifacts': 'none'}
    }
    logger.log_eeg_quality(quality_metrics)
    
    # Get session stats
    stats = logger.get_session_stats()
    print(f"Session stats: {stats}")
    
    # Validate data
    validation = logger.validate_data_integrity()
    print(f"Data validation: {validation}")
    
    # Close logger
    logger.close()
    
    print("Data logger test completed!")

if __name__ == "__main__":
    test_data_logger()