#!/usr/bin/env python3
"""
Covert Visuospatial Attention BCI Game
- FEATURES:
- Takes a break after every 30 trials.
- Arrow cue is always visible above the fixation cross.
- Ball returns to center at a constant speed.
- Personalized threshold calibration.
"""
import pygame
import sys
import time
import random
import math
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np

from eeg_stream import EEGStreamer
from eeg_processing import EEGProcessor
from data_logger import DataLogger

class VisuospatialAttentionGame:
    def __init__(self):
        pygame.init()
        self.screen_info = pygame.display.Info()
        self.screen_width = int(self.screen_info.current_w)
        self.screen_height = int(self.screen_info.current_h)
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), pygame.FULLSCREEN)
        pygame.display.set_caption("Visuospatial Attention BCI Game")

        self.BLACK, self.WHITE, self.GRAY = (0, 0, 0), (255, 255, 255), (80, 80, 80)
        self.FIXATION_COLOR = (255, 255, 255)
        self.ARROW_COLOR = (255, 255, 255)

        self.TRIAL_DURATION = 5.0
        self.TOTAL_TRIALS = 150
        self.SEMI_CIRCLE_RADIUS = 100
        self.BALL_RADIUS, self.FIXATION_SIZE = 20, 15
        self.ATTENTION_THRESHOLD = 0.1

        self.center_x, self.center_y = self.screen_width // 2, self.screen_height // 2
        self.current_trial = 0
        self.game_running = False
        self.ball_x = self.center_x
        self.active_side = 'left'
        
        self.eeg_streamer, self.eeg_processor, self.data_logger = None, None, None
        
        self.trial_asymmetry = 0
        self.trial_attention_direction = 'none'
        self.trial_alpha_o1, self.trial_alpha_o2, self.trial_alpha_p3, self.trial_alpha_p4 = 0, 0, 0, 0
        
        self.clock = pygame.time.Clock()

    def setup_participant(self):
        root = tk.Tk()
        root.withdraw()
        participant_name = simpledialog.askstring("Participant Setup", "Enter participant name:")
        if not participant_name:
            sys.exit("Participant name is required.")
        root.destroy()
        self.data_logger = DataLogger(participant_name)
        print(f"Setup complete for participant: {participant_name}")

    def initialize_eeg(self):
        try:
            self.eeg_streamer = EEGStreamer()
            self.eeg_processor = EEGProcessor()
            if not self.eeg_streamer.connect():
                return False
            self.eeg_streamer.start_streaming()
            print("EEG system initialized successfully.")
            return True
        except Exception as e:
            print(f"EEG initialization failed: {e}")
            return False

    def run_calibration(self, duration_seconds=30):
        print("\n--- Starting Calibration Phase ---")
        print(f"Please relax and look at the fixation cross for {duration_seconds} seconds.")
        
        start_time = time.time()
        asymmetry_values = []
        
        while time.time() - start_time < duration_seconds:
            self.screen.fill(self.BLACK)
            pygame.draw.line(self.screen, self.FIXATION_COLOR, (self.center_x - self.FIXATION_SIZE, self.center_y), (self.center_x + self.FIXATION_SIZE, self.center_y), 3)
            pygame.draw.line(self.screen, self.FIXATION_COLOR, (self.center_x, self.center_y - self.FIXATION_SIZE), (self.center_x, self.center_y + self.FIXATION_SIZE), 3)
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    print("\nCalibration skipped.")
                    return self.ATTENTION_THRESHOLD

            eeg_data = self.eeg_streamer.get_latest_data(n_samples=500)
            if eeg_data is not None:
                _, _, _, _, asymmetry = self.eeg_processor.process_realtime(eeg_data)
                asymmetry_values.append(asymmetry)
            
            remaining = duration_seconds - (time.time() - start_time)
            print(f"Calibrating... {remaining:.1f} seconds remaining.", end='\r')
            self.clock.tick(20)

        print("\nCalibration complete.")
        if not asymmetry_values:
            print("Warning: No asymmetry data collected. Using default threshold.")
            return self.ATTENTION_THRESHOLD
            
        absolute_asymmetries = [abs(v) for v in asymmetry_values if not np.isnan(v) and not np.isinf(v)]
        median_noise = np.median(absolute_asymmetries)
        personalized_threshold = median_noise * 2.0

        if personalized_threshold < 0.01:
            personalized_threshold = 0.01
        
        print(f"Baseline Median Noise Magnitude: {median_noise:.4f}")
        print(f"Personalized Threshold set to: {personalized_threshold:.4f}")
        time.sleep(3)
        return personalized_threshold

    def start_trial(self):
        self.current_trial += 1
        self.trial_attention_direction = 'none'
        self.trial_start_time = time.time()
        self.active_side = random.choice(['left', 'right'])
        print(f"\nTrial {self.current_trial}/{self.TOTAL_TRIALS} - Target: {self.active_side}")

    def update_ball_position(self):
        try:
            eeg_data = self.eeg_streamer.get_latest_data(n_samples=500)
            if eeg_data is not None:
                alpha_o1, alpha_o2, alpha_p3, alpha_p4, asymmetry = self.eeg_processor.process_realtime(eeg_data)
                self.trial_asymmetry = asymmetry
                self.trial_alpha_o1, self.trial_alpha_o2 = alpha_o1, alpha_o2
                self.trial_alpha_p3, self.trial_alpha_p4 = alpha_p3, alpha_p4
                
                if self.trial_asymmetry > self.ATTENTION_THRESHOLD:
                    self.trial_attention_direction = 'right'
                elif self.trial_asymmetry < -self.ATTENTION_THRESHOLD:
                    self.trial_attention_direction = 'left'
                else:
                    self.trial_attention_direction = 'center'
                
                print(f"Asymmetry: {self.trial_asymmetry:.4f}  Direction: {self.trial_attention_direction}", end='\r')
        except Exception as e:
            print(f"\nEEG processing error: {e}")

    def smooth_ball_movement(self):
        move_speed = 5
        return_speed = 2

        if self.trial_attention_direction == 'left':
            self.ball_x -= move_speed
        elif self.trial_attention_direction == 'right':
            self.ball_x += move_speed
        else:
            if self.ball_x > self.center_x:
                self.ball_x -= return_speed
                if self.ball_x < self.center_x: self.ball_x = self.center_x
            elif self.ball_x < self.center_x:
                self.ball_x += return_speed
                if self.ball_x > self.center_x: self.ball_x = self.center_x
        
        self.ball_x = max(self.BALL_RADIUS, min(self.ball_x, self.screen_width - self.BALL_RADIUS))

    def draw_cue_arrow(self):
        arrow_length = 30
        arrow_head_size = 12
        arrow_y_pos = self.center_y - 50 
        
        if self.active_side == 'left':
            start_pos = (self.center_x + arrow_length // 2, arrow_y_pos)
            end_pos = (self.center_x - arrow_length // 2, arrow_y_pos)
            pygame.draw.line(self.screen, self.ARROW_COLOR, start_pos, end_pos, 4)
            pygame.draw.polygon(self.screen, self.ARROW_COLOR, [
                end_pos,
                (end_pos[0] + arrow_head_size, end_pos[1] - arrow_head_size),
                (end_pos[0] + arrow_head_size, end_pos[1] + arrow_head_size)
            ])
        elif self.active_side == 'right':
            start_pos = (self.center_x - arrow_length // 2, arrow_y_pos)
            end_pos = (self.center_x + arrow_length // 2, arrow_y_pos)
            pygame.draw.line(self.screen, self.ARROW_COLOR, start_pos, end_pos, 4)
            pygame.draw.polygon(self.screen, self.ARROW_COLOR, [
                end_pos,
                (end_pos[0] - arrow_head_size, end_pos[1] - arrow_head_size),
                (end_pos[0] - arrow_head_size, end_pos[1] + arrow_head_size)
            ])

    def draw_elements(self):
        self.screen.fill(self.BLACK)
        
        arc_outline_width = 4
        
        left_rect = pygame.Rect(0 - self.SEMI_CIRCLE_RADIUS, self.center_y - self.SEMI_CIRCLE_RADIUS, self.SEMI_CIRCLE_RADIUS * 2, self.SEMI_CIRCLE_RADIUS * 2)
        right_rect = pygame.Rect(self.screen_width - self.SEMI_CIRCLE_RADIUS, self.center_y - self.SEMI_CIRCLE_RADIUS, self.SEMI_CIRCLE_RADIUS * 2, self.SEMI_CIRCLE_RADIUS * 2)
        
        pygame.draw.arc(self.screen, self.GRAY, left_rect, -math.pi/2, math.pi/2, arc_outline_width)
        pygame.draw.arc(self.screen, self.GRAY, right_rect, math.pi/2, 3*math.pi/2, arc_outline_width)
        
        pygame.draw.line(self.screen, self.FIXATION_COLOR, (self.center_x - self.FIXATION_SIZE, self.center_y), (self.center_x + self.FIXATION_SIZE, self.center_y), 3)
        pygame.draw.line(self.screen, self.FIXATION_COLOR, (self.center_x, self.center_y - self.FIXATION_SIZE), (self.center_x, self.center_y + self.FIXATION_SIZE), 3)

        self.draw_cue_arrow()

        pygame.draw.circle(self.screen, self.WHITE, (int(self.ball_x), int(self.center_y)), self.BALL_RADIUS)
        
        pygame.display.flip()

    def end_trial(self):
        correct = (self.active_side == self.trial_attention_direction)
        print(f"\nTrial {self.current_trial} End: Target={self.active_side}, Detected={self.trial_attention_direction}, Correct={correct}")
        if self.data_logger:
            self.data_logger.log_trial(
                trial_number=self.current_trial, timestamp=datetime.now(), target_side=self.active_side,
                alpha_o1=self.trial_alpha_o1, alpha_o2=self.trial_alpha_o2, 
                alpha_p3=self.trial_alpha_p3, alpha_p4=self.trial_alpha_p4, 
                alpha_asymmetry=self.trial_asymmetry,
                attention_direction=self.trial_attention_direction, correct=1 if correct else 0,
                reaction_time=0.5, confidence=abs(self.trial_asymmetry))

    # --- NEW FUNCTION FOR THE BREAK SCREEN ---
    def run_break_screen(self):
        print("\n--- Break Time ---")
        break_active = True
        
        font_large = pygame.font.Font(None, 72)
        font_small = pygame.font.Font(None, 48)

        break_text = font_large.render("Break Time!", True, self.WHITE)
        continue_text = font_small.render("Press SPACE to continue...", True, self.GRAY)
        
        break_rect = break_text.get_rect(center=(self.center_x, self.center_y - 50))
        continue_rect = continue_text.get_rect(center=(self.center_x, self.center_y + 50))

        while break_active:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.game_running = False
                    break_active = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        break_active = False # Exit the break loop
                    if event.key == pygame.K_ESCAPE:
                        self.game_running = False
                        break_active = False

            self.screen.fill(self.BLACK)
            self.screen.blit(break_text, break_rect)
            self.screen.blit(continue_text, continue_rect)
            pygame.display.flip()
            self.clock.tick(15)

    def run_game(self):
        print("\nStarting BCI Game...")
        self.game_running = True
        self.start_trial()
        while self.game_running:
            current_time = time.time()
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.game_running = False
            
            self.update_ball_position()
            self.smooth_ball_movement()
            self.draw_elements()

            if current_time - self.trial_start_time > self.TRIAL_DURATION:
                self.end_trial()
                
                # --- MODIFIED: ADDED BREAK LOGIC ---
                if self.current_trial % 30 == 0 and self.current_trial < self.TOTAL_TRIALS:
                    self.run_break_screen()
                # -----------------------------------

                if self.current_trial >= self.TOTAL_TRIALS:
                    self.game_running = False
                else:
                    if self.game_running: # Check if user quit during break
                        self.start_trial()
            
            self.clock.tick(60)
            
        print("Game loop finished.")
        if self.data_logger: self.data_logger.close()
        if self.eeg_streamer: self.eeg_streamer.disconnect()
        pygame.quit()

def main():
    try:
        game = VisuospatialAttentionGame()
        game.setup_participant()
        if game.initialize_eeg():
            
            print("\nEEG system warming up, please wait...")
            time.sleep(2) 
            
            personalized_threshold = game.run_calibration(duration_seconds=30)
            game.ATTENTION_THRESHOLD = personalized_threshold
            
            game.run_game()
            
        else:
            print("Failed to initialize EEG. Exiting.")
            
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()