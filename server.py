#!/usr/bin/env python3
"""
PID Controller Dashboard Backend
Flask server for motor control and monitoring
"""

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import json
import numpy as np
from datetime import datetime
import logging
import socket
from typing import Dict, Optional, List
import queue
import traceback
import os
from collections import deque
import random
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import csv
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArduinoPIDController:
    def __init__(self, arduino_ip: str, arduino_port: int = 80, timeout: float = 1.0):  # Reduced timeout for lower latency
        self.arduino_ip = arduino_ip
        self.arduino_port = arduino_port
        self.timeout = timeout
        self.socket = None
        self.connected = False
        self.status_data = {}
        self.running = True
        self.last_heartbeat = 0
        self.last_status_log = 0  # Rate limiting for verbose logging
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 5
        self.last_reconnect_attempt = 0
        self.reconnect_interval = 5.0  # Seconds between reconnect attempts
        
    def connect(self) -> bool:
        """Connect to Arduino over WiFi"""
        try:
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect((self.arduino_ip, self.arduino_port))
            
            response = self._read_response()
            logger.info(f"Arduino connection response: {response}")
            
            self.connected = True
            self.last_heartbeat = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        self.running = False
        self.connected = False
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        logger.info("Disconnected from Arduino")
    
    def send_command(self, command: str) -> str:
        """Send command to Arduino and return response"""
        if not self.connected:
            return "ERROR Not connected"
        
        try:
            self.socket.send((command + '\n').encode('utf-8'))
            response = self._read_response()
            self.last_heartbeat = time.time()
            return response
            
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            self.connected = False
            return f"ERROR {e}"
    
    def _read_response(self) -> str:
        """Read response from Arduino"""
        try:
            data = self.socket.recv(4096).decode('utf-8', errors='ignore').strip()
            
            if 'STATUS ' in data:
                status_lines = data.split('STATUS ')
                
                for line in reversed(status_lines):
                    if line.strip() and ',' in line:
                        return 'STATUS ' + line.strip().split('\n')[0]
            
            return data
            
        except Exception as e:
            return f"ERROR {e}"
    
    def get_status(self) -> Dict:
        """Get current status"""
        try:
            response = self.send_command("STATUS")
            
            if 'STATUS ' in response:
                status_line = response
                if response.count('STATUS ') > 1:
                    lines = response.split('\n')
                    for line in lines:
                        if line.startswith('STATUS ') and ',' in line:
                            status_line = line
                            break
                
                status_data = self._parse_status(status_line[7:])
                # Reduce logging verbosity - only log occasionally
                if time.time() - self.last_status_log > 5.0:
                    logger.info(f"Final parsed status: {status_data}")
                    self.last_status_log = time.time()
                return status_data
            else:
                logger.warning(f"No STATUS found in response: '{response[:100]}...'")
                return {}
                
        except Exception as e:
            logger.error(f"Status error: {e}")
            return {}
    
    def _parse_status(self, status_string: str) -> Dict:
        """Parse status string"""
        try:
            first_line = status_string.split('\n')[0].strip()
            
            pairs = first_line.split(',')
            status_dict = {}
            
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    if value.lower() == 'true':
                        value = True
                    elif value.lower() == 'false':
                        value = False
                    else:
                        try:
                            if '.' in value:
                                value = float(value)
                            else:
                                value = int(value)
                        except ValueError:
                            pass
                    
                    status_dict[key] = value
            
            status_dict['timestamp'] = time.time()
            return status_dict
            
        except Exception as e:
            logger.error(f"Status parsing error: {e}")
            return {}
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        return self.connected and (time.time() - self.last_heartbeat < 5.0)  # Reduced timeout for faster detection
    
    def should_reconnect(self) -> bool:
        """Check if we should attempt reconnection"""
        if self.connected:
            return False
        current_time = time.time()
        return (current_time - self.last_reconnect_attempt > self.reconnect_interval and 
                self.reconnect_attempts < self.max_reconnect_attempts)
    
    def auto_reconnect(self) -> bool:
        """Attempt automatic reconnection"""
        if not self.should_reconnect():
            return False
        
        self.last_reconnect_attempt = time.time()
        self.reconnect_attempts += 1
        
        logger.info(f"Auto-reconnect attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")
        
        if self.connect():
            self.reconnect_attempts = 0  # Reset on successful connection
            return True
        
        return False

class DataLogger:
    """Comprehensive data logging system"""
    def __init__(self):
        self.session_id = None
        self.session_data = []
        self.training_data = []
        self.reward_history = []
        self.pid_history = []
        self.ensure_directories()
    
    def ensure_directories(self):
        """Create necessary directories"""
        os.makedirs('data', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('training_sessions', exist_ok=True)
    
    def start_session(self, session_type="manual"):
        """Start a new logging session"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{session_type}_{timestamp}"
        self.session_data = []
        self.training_data = []
        self.reward_history = []
        self.pid_history = []
        logger.info(f"Started logging session: {self.session_id}")
    
    def log_status(self, status_data):
        """Log status data point"""
        if self.session_id and status_data:
            entry = {
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                **status_data
            }
            self.session_data.append(entry)
    
    def log_training_episode(self, episode, params, reward, target, duration):
        """Log training episode data"""
        if self.session_id:
            entry = {
                'episode': episode,
                'timestamp': time.time(),
                'datetime': datetime.now().isoformat(),
                'kp': params['kp'],
                'ki': params['ki'],
                'kd': params['kd'],
                'reward': reward,
                'target': target,
                'duration': duration
            }
            self.training_data.append(entry)
            self.reward_history.append(reward)
            self.pid_history.append(params.copy())
    
    def save_session(self, save_format='json'):
        """Save session data to files"""
        if not self.session_id:
            return None
        
        session_dir = f"training_sessions/{self.session_id}"
        os.makedirs(session_dir, exist_ok=True)
        
        files_created = []
        
        # Save as JSON
        if save_format in ['json', 'both']:
            json_file = f"{session_dir}/session_data.json"
            with open(json_file, 'w') as f:
                json.dump({
                    'session_id': self.session_id,
                    'status_data': self.session_data,
                    'training_data': self.training_data,
                    'reward_history': self.reward_history,
                    'pid_history': self.pid_history
                }, f, indent=2)
            files_created.append(json_file)
        
        # Save as CSV
        if save_format in ['csv', 'both']:
            if self.training_data:
                csv_file = f"{session_dir}/training_data.csv"
                pd.DataFrame(self.training_data).to_csv(csv_file, index=False)
                files_created.append(csv_file)
            
            if self.session_data:
                status_csv = f"{session_dir}/status_data.csv"
                pd.DataFrame(self.session_data).to_csv(status_csv, index=False)
                files_created.append(status_csv)
        
        return files_created

    def create_visualizations(self):
        """Create matplotlib visualizations"""
        if not self.session_id or not self.training_data:
            return []
        
        session_dir = f"training_sessions/{self.session_id}"
        plots_created = []
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
        # 1. Reward Progress Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        episodes = [d['episode'] for d in self.training_data]
        rewards = [d['reward'] for d in self.training_data]
        
        ax.plot(episodes, rewards, 'b-', linewidth=2, label='Episode Reward')
        ax.plot(episodes, pd.Series(rewards).rolling(window=5).mean(), 'r-', linewidth=3, label='5-Episode Moving Average')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Progress - Reward vs Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        reward_plot = f"{session_dir}/reward_progress.png"
        fig.savefig(reward_plot, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plots_created.append(reward_plot)
        
        # 2. PID Parameters Evolution
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        kp_values = [d['kp'] for d in self.training_data]
        ki_values = [d['ki'] for d in self.training_data]
        kd_values = [d['kd'] for d in self.training_data]
        
        ax1.plot(episodes, kp_values, 'g-', linewidth=2, label='Kp')
        ax1.set_ylabel('Kp Value')
        ax1.set_title('PID Parameter Evolution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(episodes, ki_values, 'orange', linewidth=2, label='Ki')
        ax2.set_ylabel('Ki Value')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        ax3.plot(episodes, kd_values, 'purple', linewidth=2, label='Kd')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Kd Value')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        pid_plot = f"{session_dir}/pid_evolution.png"
        fig.savefig(pid_plot, dpi=300, bbox_inches='tight')
        plt.close(fig)
        plots_created.append(pid_plot)
        
        # 3. Reward vs PID Parameters Correlation
        if len(self.training_data) > 10:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Reward vs Kp
            ax1.scatter(kp_values, rewards, alpha=0.6, c=episodes, cmap='viridis')
            ax1.set_xlabel('Kp')
            ax1.set_ylabel('Reward')
            ax1.set_title('Reward vs Kp')
            ax1.grid(True, alpha=0.3)
            
            # Reward vs Ki
            ax2.scatter(ki_values, rewards, alpha=0.6, c=episodes, cmap='viridis')
            ax2.set_xlabel('Ki')
            ax2.set_ylabel('Reward')
            ax2.set_title('Reward vs Ki')
            ax2.grid(True, alpha=0.3)
            
            # Reward vs Kd
            ax3.scatter(kd_values, rewards, alpha=0.6, c=episodes, cmap='viridis')
            ax3.set_xlabel('Kd')
            ax3.set_ylabel('Reward')
            ax3.set_title('Reward vs Kd')
            ax3.grid(True, alpha=0.3)
            
            # Best parameters highlight
            best_idx = rewards.index(max(rewards))
            best_kp, best_ki, best_kd = kp_values[best_idx], ki_values[best_idx], kd_values[best_idx]
            
            ax4.text(0.1, 0.8, f'Best Parameters:', fontsize=14, fontweight='bold', transform=ax4.transAxes)
            ax4.text(0.1, 0.7, f'Kp: {best_kp:.3f}', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.1, 0.6, f'Ki: {best_ki:.3f}', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.1, 0.5, f'Kd: {best_kd:.3f}', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.1, 0.4, f'Reward: {max(rewards):.2f}', fontsize=12, transform=ax4.transAxes)
            ax4.text(0.1, 0.3, f'Episode: {episodes[best_idx]}', fontsize=12, transform=ax4.transAxes)
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.set_title('Best Results Summary')
            ax4.axis('off')
            
            correlation_plot = f"{session_dir}/reward_correlation.png"
            fig.savefig(correlation_plot, dpi=300, bbox_inches='tight')
            plt.close(fig)
            plots_created.append(correlation_plot)
        
        return plots_created

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'pid-controller-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

class PPOPIDOptimizer:
    """PPO-inspired PID parameter optimizer with safety constraints"""
    
    def __init__(self, baseline_params=None, learning_rate=0.05, exploration_rate=0.3, conservative_mode=True):
        # Initialize with baseline or default parameters
        if baseline_params:
            self.current_params = baseline_params.copy()
        else:
            self.current_params = {'kp': 1.0, 'ki': 0.0, 'kd': 0.0}
            
        # Store learning settings
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.conservative_mode = conservative_mode
        
        # PPO hyperparameters
        self.clip_epsilon = 0.2
        self.entropy_coeff = 0.01
        
        # Safety constraints
        self.param_bounds = {
            'kp': (0.1, 50.0),  # Proportional gain bounds
            'ki': (0.0, 10.0),  # Integral gain bounds  
            'kd': (0.0, 20.0)   # Derivative gain bounds
        }
        
        # Experience buffer
        self.experience_buffer = deque(maxlen=100)
        self.best_params = self.current_params.copy()
        self.best_reward = -float('inf')
        
        # Performance tracking
        self.episode_rewards = deque(maxlen=20)
        self.param_history = deque(maxlen=50)
        
    def clip_params(self, params):
        """Ensure parameters stay within safe bounds"""
        clipped = {}
        for key, value in params.items():
            min_val, max_val = self.param_bounds[key]
            clipped[key] = max(min_val, min(max_val, value))
        return clipped
    
    def generate_safe_variation(self, base_params, std_scale=None):
        """Generate parameter variation with safety constraints"""
        new_params = {}
        
        # Use conservative_mode to determine variation scale
        if std_scale is None:
            if self.conservative_mode:
                std_scale = 0.05  # Very small variations (5% of range)
            else:
                std_scale = 0.15  # Moderate variations (15% of range)
        
        for key, value in base_params.items():
            # For conservative mode, vary based on current value rather than full range
            if self.conservative_mode and value > 0:
                # Vary by percentage of current value (much smaller changes)
                std_dev = value * std_scale * 2  # 10% of current value
            else:
                # Traditional approach - vary by percentage of parameter bounds
                min_val, max_val = self.param_bounds[key]
                param_range = max_val - min_val
                std_dev = param_range * std_scale
            
            # Generate variation with clipping
            variation = np.random.normal(0, std_dev)
            new_value = value + variation
            
            # Ensure within bounds
            min_val, max_val = self.param_bounds[key]
            new_params[key] = max(min_val, min(max_val, new_value))
            
        return new_params
    
    def update_policy(self, params, reward, episode):
        """PPO-inspired policy update with safety features"""
        # Store experience
        self.experience_buffer.append({
            'params': params.copy(),
            'reward': reward,
            'episode': episode
        })
        
        self.episode_rewards.append(reward)
        self.param_history.append(params.copy())
        
        # Update best parameters
        if reward > self.best_reward:
            self.best_reward = reward
            self.best_params = params.copy()
            logger.info(f"New best parameters: {self.best_params} with reward {reward:.2f}")
        
        # PPO-style update every few episodes
        if len(self.experience_buffer) >= 10 and episode % 5 == 0:
            self._ppo_update()
    
    def _ppo_update(self):
        """Simplified PPO policy update"""
        if len(self.experience_buffer) < 10:
            return
            
        # Get recent experiences
        recent_experiences = list(self.experience_buffer)[-10:]
        rewards = [exp['reward'] for exp in recent_experiences]
        
        # Normalize rewards
        if len(rewards) > 1:
            reward_mean = np.mean(rewards)
            reward_std = np.std(rewards) + 1e-8
            normalized_rewards = [(r - reward_mean) / reward_std for r in rewards]
        else:
            normalized_rewards = rewards
        
        # Update current parameters towards better performing ones
        if len(recent_experiences) >= 3:
            # Find top 30% performers
            sorted_exp = sorted(recent_experiences, key=lambda x: x['reward'], reverse=True)
            top_performers = sorted_exp[:max(1, len(sorted_exp) // 3)]
            
            # Weighted average of top performers
            total_weight = sum(exp['reward'] for exp in top_performers)
            if total_weight > 0:
                new_params = {'kp': 0, 'ki': 0, 'kd': 0}
                for exp in top_performers:
                    weight = exp['reward'] / total_weight
                    for key in new_params:
                        new_params[key] += weight * exp['params'][key]
                
                # Smooth update towards better parameters (conservative update)
                update_rate = self.learning_rate
                for key in self.current_params:
                    self.current_params[key] = (
                        (1.0 - update_rate) * self.current_params[key] + 
                        update_rate * new_params[key]
                    )
                
                # Ensure parameters stay within bounds
                self.current_params = self.clip_params(self.current_params)
    
    def get_next_params(self, episode):
        """Get next parameter set to test"""
        if episode < 5:
            # Initial exploration phase - test around baseline with conservative variations
            if self.conservative_mode:
                return self.generate_safe_variation(self.current_params, std_scale=0.08)  # 8% variation
            else:
                return self.generate_safe_variation(self.current_params, std_scale=0.2)
        elif episode % 8 == 0 and len(self.experience_buffer) > 10:
            # Periodically test best known parameters with small variation
            return self.generate_safe_variation(self.best_params)  # Use default conservative scaling
        else:
            # Normal exploration around current policy
            return self.generate_safe_variation(self.current_params)  # Use default conservative scaling
    
    def get_training_stats(self):
        """Get detailed training statistics"""
        stats = {
            'current_params': self.current_params.copy(),
            'best_params': self.best_params.copy(),
            'best_reward': self.best_reward,
            'episodes_completed': len(self.param_history),
            'avg_reward_last_10': 0,
            'param_stability': 0,
            'exploration_rate': 0
        }
        
        if len(self.episode_rewards) > 0:
            recent_rewards = list(self.episode_rewards)[-10:]
            stats['avg_reward_last_10'] = np.mean(recent_rewards)
            
            # Calculate parameter stability (lower = more stable)
            if len(self.param_history) >= 5:
                recent_params = list(self.param_history)[-5:]
                kp_values = [p['kp'] for p in recent_params]
                stats['param_stability'] = np.std(kp_values)
                
            # Estimate exploration rate
            if len(recent_rewards) > 1:
                stats['exploration_rate'] = np.std(recent_rewards) / (np.mean(recent_rewards) + 1e-8)
        
        return stats

# Global variables
arduino_controller = None
training_active = False
ppo_optimizer = None
data_logger = DataLogger()
training_data = {
    'active': False,
    'episode': 0,
    'total_episodes': 0,
    'best_reward': -float('inf'),
    'best_params': None,
    'completed': False
}
status_monitoring_active = False

# Customizable reward function parameters
reward_config = {
    'stability_weight': 40,        # 0-100: How much to value stability
    'accuracy_weight': 20,         # 0-100: How much to value accuracy  
    'cumulative_weight': 15,       # 0-100: How much to penalize cumulative error
    'settling_weight': 15,         # 0-100: How much to value settling performance
    'settling_time_weight': 10,    # 0-100: How much to value fast settling time
    'overshoot_penalty': 5.0,      # 0-10: Multiplier for overshoot penalty
    'stability_sensitivity': 3.0,   # 0-10: How sensitive to movement changes
    'aggressive_penalty': 0.3,      # 0-1: Penalty for aggressive parameters
    'conservative_bonus': 0.2,      # 0-1: Bonus for stable non-overshoot behavior
    'settling_threshold': 3.0,      # 0-10: Error threshold for "settled" (degrees)
    'settling_window': 1.0          # 0-5: Time window that must stay within threshold (seconds)
}

def get_local_ip():
    """Get local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

def create_dashboard_template():
    """Create the dashboard template file"""
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PID Controller Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: system-ui, sans-serif; background: linear-gradient(135deg, #1e293b 0%, #334155 100%); min-height: 100vh; color: #f1f5f9; padding: 20px; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { text-align: center; margin-bottom: 30px; }
        .header h1 { font-size: 2.5rem; font-weight: 700; color: #60a5fa; }
        .card { background: rgba(30, 41, 59, 0.8); border-radius: 15px; padding: 20px; border: 1px solid rgba(59, 130, 246, 0.2); margin-bottom: 20px; }
        .card-title { font-size: 1.2rem; font-weight: 600; margin-bottom: 15px; color: #60a5fa; border-bottom: 2px solid rgba(59, 130, 246, 0.3); padding-bottom: 8px; }
        .button { background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%); color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; margin: 3px; transition: all 0.3s ease; }
        .button:hover { transform: translateY(-1px); }
        .button.danger { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
        .button.success { background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); }
        .button.warning { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-indicator.connected { background: #22c55e; box-shadow: 0 0 10px rgba(34, 197, 94, 0.5); }
        .status-indicator.disconnected { background: #ef4444; }
        .input-group { margin-bottom: 12px; }
        .input-group input { width: 100%; padding: 8px 12px; border: 1px solid #475569; border-radius: 6px; background: rgba(15, 23, 42, 0.6); color: #f1f5f9; }
        .status-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        .status-item { background: rgba(15, 23, 42, 0.5); padding: 12px; border-radius: 8px; text-align: center; }
        .status-value { font-size: 1.4rem; font-weight: bold; color: #60a5fa; font-family: monospace; }
        .status-label { font-size: 0.8rem; color: #94a3b8; margin-top: 4px; }
        .chart-container { height: 300px; background: rgba(15, 23, 42, 0.5); border-radius: 10px; padding: 15px; }
        .dial { width: 150px; height: 150px; border: 3px solid #475569; border-radius: 50%; position: relative; background: radial-gradient(circle, #0f172a 0%, #1e293b 100%); margin: 0 auto; }
        .dial-pointer { position: absolute; top: 50%; left: 50%; width: 3px; height: 60px; background: #60a5fa; transform-origin: bottom center; transform: translate(-50%, -100%) rotate(0deg); transition: transform 0.5s ease; }
        .dial-center { position: absolute; top: 50%; left: 50%; width: 12px; height: 12px; background: #60a5fa; border-radius: 50%; transform: translate(-50%, -50%); }
        .dial-target { position: absolute; top: 50%; left: 50%; width: 2px; height: 50px; background: #ef4444; transform-origin: bottom center; transform: translate(-50%, -100%) rotate(0deg); }
        .log-container { background: rgba(15, 23, 42, 0.8); border-radius: 10px; padding: 15px; height: 250px; overflow-y: auto; font-family: monospace; font-size: 0.85rem; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>PID Controller Dashboard</h1>
            <p>Real-time motor control and system monitoring</p>
        </div>

        <!-- Connection -->
        <div class="card">
            <div class="card-title">Connection Status</div>
            <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap;">
                <div class="status-indicator disconnected" id="connectionIndicator"></div>
                <span id="connectionText">Disconnected</span>
                <input type="text" id="arduinoIP" placeholder="Arduino IP" value="192.168.0.224" style="width: 150px;">
                <button class="button" onclick="connectArduino()" id="connectBtn">Connect</button>
                <div id="lastUpdate" style="font-size: 0.8rem; color: #94a3b8;">Last: Never</div>
            </div>
            
            <div style="margin-top: 15px; padding-top: 15px; border-top: 1px solid rgba(59, 130, 246, 0.2);">
                <div style="font-size: 0.9rem; color: #94a3b8; margin-bottom: 10px;">Debug Tests:</div>
                <button class="button warning" onclick="pingTest()">Ping Test</button>
                <button class="button warning" onclick="requestStatus()">Request Status</button>
                <button class="button warning" onclick="testUIDirectly()">Test UI Direct</button>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
            <!-- Status -->
            <div class="card">
                <div class="card-title">System Status</div>
                <div class="status-grid">
                    <div class="status-item">
                        <div class="status-value" id="currentDegrees">0.0&deg;</div>
                        <div class="status-label">Current Position</div>
                    </div>
                    <div class="status-item">
                        <div class="status-value" id="targetDegrees">0.0&deg;</div>
                        <div class="status-label">Target Position</div>
                    </div>
                    <div class="status-item">
                        <div class="status-value" id="pidStatus">OFF</div>
                        <div class="status-label">PID Control</div>
                    </div>
                    <div class="status-item">
                        <div class="status-value" id="motorStatus">STOPPED</div>
                        <div class="status-label">Motor Status</div>
                    </div>
                </div>
            </div>

            <!-- Control -->
            <div class="card">
                <div class="card-title">Manual Control</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
                    <button class="button success" onclick="sendCommand('FORWARD')">Forward</button>
                    <button class="button danger" onclick="sendCommand('STOP')">Stop</button>
                    <button class="button" onclick="sendCommand('REVERSE')">Reverse</button>
                    <button class="button" onclick="sendCommand('ZERO')">Zero</button>
                </div>
                <div class="input-group">
                    <label>Target Position (degrees)</label>
                    <input type="number" id="targetInput" placeholder="Enter target in degrees" value="90" step="0.1" min="0" max="360">
                    <button class="button" onclick="setTarget()">Set Target</button>
                </div>
                
                <div class="input-group">
                    <label>Quick Targets</label>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 5px;">
                        <button class="button" onclick="setQuickTarget(0)">0&deg;</button>
                        <button class="button" onclick="setQuickTarget(90)">90&deg;</button>
                        <button class="button" onclick="setQuickTarget(180)">180&deg;</button>
                        <button class="button" onclick="setQuickTarget(270)">270&deg;</button>
                    </div>
                </div>
                
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 10px; text-align: center;">
                    Press ENTER or SPACE for emergency stop
                </div>
            </div>

            <!-- PID -->
            <div class="card">
                <div class="card-title">PID Control</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                    <div class="input-group">
                        <label>Kp</label>
                        <input type="number" id="kpValue" step="0.1" value="1.0">
                    </div>
                    <div class="input-group">
                        <label>Ki</label>
                        <input type="number" id="kiValue" step="0.01" value="0.0">
                    </div>
                    <div class="input-group">
                        <label>Kd</label>
                        <input type="number" id="kdValue" step="0.1" value="0.0">
                    </div>
                </div>
                <button class="button" onclick="updatePID()">Update PID</button>
                <div style="margin-top: 10px;">
                    <button class="button success" onclick="sendCommand('PID_ON')">Enable</button>
                    <button class="button danger" onclick="sendCommand('PID_OFF')">Disable</button>
                </div>
            </div>
        </div>

        <!-- Visualization -->
        <div class="card">
            <div class="card-title">Position Visualization</div>
            <div style="display: grid; grid-template-columns: 1fr 2fr; gap: 20px;">
                <div style="text-align: center;">
                    <div class="dial">
                        <div class="dial-pointer" id="dialPointer"></div>
                        <div class="dial-target" id="dialTarget"></div>
                        <div class="dial-center"></div>
                    </div>
                    <div style="margin-top: 15px;">
                        Current: <span id="dialCurrent">0.0&deg;</span><br>
                        Target: <span id="dialTargetText">0.0&deg;</span>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="positionChart"></canvas>
                </div>
            </div>
        </div>

        <!-- RL Training -->
        <div class="card">
            <div class="card-title">RL Training</div>
            <div style="display: grid; grid-template-columns: 1fr 2fr 1fr; gap: 20px;">
                <div>
                    <div class="input-group">
                        <label>Episodes</label>
                        <input type="number" id="episodes" value="20">
                    </div>
                    <div class="input-group">
                        <label>Duration (s)</label>
                        <input type="number" id="duration" value="8">
                    </div>
                    <button class="button success" onclick="startTraining()">Start Training</button>
                    <button class="button danger" onclick="stopTraining()">Stop</button>
                </div>
                <div id="trainingStatus" style="text-align: center; color: #94a3b8;">
                    Ready to train
                </div>
                <div style="text-align: center;">
                    <div>Best Reward: <span id="bestReward">--</span></div>
                    <div>Episode: <span id="currentEpisode">0</span></div>
                </div>
            </div>
        </div>

        <!-- Log -->
        <div class="log-container">
            <div style="color: #60a5fa; margin-bottom: 10px;">System Log</div>
            <div id="logContent">Ready to connect...</div>
        </div>
    </div>

    <script>
        const socket = io();
        let connected = false;
        let positionChart;
        let chartData = [];
        let currentAngle = 0;
        let targetAngle = 0;
        let continuousCurrentAngle = 0;
        let continuousTargetAngle = 0;
        let lastRawCurrentAngle = 0;
        let lastRawTargetAngle = 0;

        function wrapAngleContinuous(newAngle, lastAngle, continuousAngle) {
            // Handle angle wrapping to maintain continuity
            let diff = newAngle - lastAngle;
            
            // If difference is greater than 180°, we wrapped from 360 to 0
            if (diff > 180) {
                diff -= 360;
            }
            // If difference is less than -180°, we wrapped from 0 to 360
            else if (diff < -180) {
                diff += 360;
            }
            
            return continuousAngle + diff;
        }

        function initChart() {
            const ctx = document.getElementById('positionChart').getContext('2d');
            positionChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Position',
                        data: [],
                        borderColor: '#60a5fa',
                        tension: 0.4,
                        pointRadius: 0
                    }, {
                        label: 'Target',
                        data: [],
                        borderColor: '#ef4444',
                        borderDash: [5, 5],
                        pointRadius: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { labels: { color: '#cbd5e1' } } },
                    scales: {
                        x: { display: false },
                        y: { 
                            min: 0, 
                            max: 360, 
                            ticks: { 
                                color: '#94a3b8',
                                stepSize: 90
                            },
                            title: {
                                display: true,
                                text: 'Angle (°)',
                                color: '#94a3b8'
                            }
                        }
                    },
                    animation: false
                }
            });
        }

        function pingTest() {
            console.log('Sending ping to server...');
            logMessage('Sending ping to server...', 'command');
            socket.emit('ping_test');
        }

        function requestStatus() {
            console.log('Requesting manual status update...');
            logMessage('Requesting manual status update...', 'command');
            socket.emit('request_status');
        }

        function testUIDirectly() {
            console.log('Testing UI update directly...');
            logMessage('Testing UI update directly...', 'warning');
            
            currentAngle = 123.4;
            targetAngle = 90.0;
            continuousCurrentAngle = 123.4;
            continuousTargetAngle = 90.0;
            lastRawCurrentAngle = 123.4;
            lastRawTargetAngle = 90.0;
            
            document.getElementById('currentDegrees').textContent = '123.4°';
            document.getElementById('targetDegrees').textContent = '90.0°';
            document.getElementById('pidStatus').textContent = 'ON';
            document.getElementById('connectionIndicator').className = 'status-indicator connected';
            document.getElementById('connectionText').textContent = 'Connected (TEST)';
            
            updateChart();
            updateDial();
            
            logMessage('UI updated directly - if you see changes, the UI works!', 'success');
        }

        socket.on('connect', () => {
            console.log('WebSocket connected to server');
            logMessage('WebSocket connected to server', 'success');
        });

        socket.on('disconnect', () => {
            console.log('WebSocket disconnected from server');
            logMessage('WebSocket disconnected from server', 'error');
        });

        socket.on('pong_test', (data) => {
            console.log('Pong received:', data);
            logMessage('Pong received: ' + data.message, 'success');
        });

        socket.on('arduino_status', (data) => {
            console.log('RECEIVED arduino_status:', data);
            logMessage('WebSocket status received: connected=' + data.connected + ', degrees=' + data.degrees, 'info');
            updateStatus(data);
        });

        socket.on('log_message', (data) => {
            logMessage(data.message, data.type);
        });

        socket.on('training_update', (data) => {
            updateTraining(data);
        });

        socket.on('training_complete', (data) => {
            logMessage('Training complete! Episodes: ' + data.episodes_completed + ', Best reward: ' + data.best_reward.toFixed(2), 'response');
            
            if (data.best_params) {
                document.getElementById('kpValue').value = data.best_params.kp.toFixed(3);
                document.getElementById('kiValue').value = data.best_params.ki.toFixed(3);
                document.getElementById('kdValue').value = data.best_params.kd.toFixed(3);
                
                logMessage('Best parameters loaded into PID controls - click "Update PID" to apply', 'response');
            }
            
            document.getElementById('trainingStatus').innerHTML = `
                <div style="color: #22c55e;">Training Complete!</div>
                <div style="font-size: 0.9rem; margin-top: 5px;">
                    Best: Kp=${data.best_params.kp.toFixed(3)}, 
                    Ki=${data.best_params.ki.toFixed(3)}, 
                    Kd=${data.best_params.kd.toFixed(3)}
                </div>
                <div style="font-size: 0.8rem; margin-top: 5px;">
                    Reward: ${data.best_reward.toFixed(2)}
                </div>
            `;
        });

        function connectArduino() {
            const ip = document.getElementById('arduinoIP').value;
            socket.emit('connect_arduino', {ip: ip});
        }

        function sendCommand(command) {
            socket.emit('send_command', {command: command});
        }

        function updatePID() {
            const kp = document.getElementById('kpValue').value;
            const ki = document.getElementById('kiValue').value;
            const kd = document.getElementById('kdValue').value;
            sendCommand('PID ' + kp + ' ' + ki + ' ' + kd);
        }

        function setTarget() {
            const target = parseFloat(document.getElementById('targetInput').value);
            sendCommand('TARGET ' + target);
        }

        function setQuickTarget(degrees) {
            sendCommand('TARGET ' + degrees);
        }

        document.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                emergencyStop();
            }
        });

        function emergencyStop() {
            sendCommand('STOP');
            logMessage('EMERGENCY STOP activated by keyboard', 'warning');
        }

        function startTraining() {
            const episodes = document.getElementById('episodes').value;
            const duration = document.getElementById('duration').value;
            socket.emit('start_rl_training', {
                episodes: parseInt(episodes),
                duration: parseFloat(duration),
                target: 180
            });
        }

        function stopTraining() {
            socket.emit('stop_rl_training');
        }

        function updateStatus(data) {
            if (!data) return;
            
            connected = data.connected || false;
            
            const indicator = document.getElementById('connectionIndicator');
            const text = document.getElementById('connectionText');
            
            if (connected) {
                indicator.className = 'status-indicator connected';
                text.textContent = 'Connected';
                
                currentAngle = data.degrees || 0;
                targetAngle = data.target_degrees || 0;
                
                // Update continuous angles for smooth charting
                continuousCurrentAngle = wrapAngleContinuous(currentAngle, lastRawCurrentAngle, continuousCurrentAngle);
                continuousTargetAngle = wrapAngleContinuous(targetAngle, lastRawTargetAngle, continuousTargetAngle);
                
                lastRawCurrentAngle = currentAngle;
                lastRawTargetAngle = targetAngle;
                
                document.getElementById('currentDegrees').textContent = currentAngle.toFixed(1) + '°';
                document.getElementById('targetDegrees').textContent = targetAngle.toFixed(1) + '°';
                document.getElementById('pidStatus').textContent = data.pid_enabled ? 'ON' : 'OFF';
                document.getElementById('motorStatus').textContent = data.pid_enabled ? 'RUNNING' : 'STOPPED';
                
                updateChart();
                updateDial();
                
                document.getElementById('lastUpdate').textContent = 'Last: ' + new Date().toLocaleTimeString();
            } else {
                indicator.className = 'status-indicator disconnected';
                text.textContent = 'Disconnected';
            }
        }

        function updateChart() {
            const now = new Date().toLocaleTimeString();
            chartData.push({time: now, current: currentAngle, target: targetAngle});
            
            if (chartData.length > 50) chartData.shift();
            
            positionChart.data.labels = chartData.map(d => d.time);
            positionChart.data.datasets[0].data = chartData.map(d => d.current);
            positionChart.data.datasets[1].data = chartData.map(d => d.target);
            positionChart.update('none');
        }

        function updateDial() {
            // Use continuous angles for smooth dial rotation
            document.getElementById('dialPointer').style.transform = 
                `translate(-50%, -100%) rotate(${continuousCurrentAngle}deg)`;
            document.getElementById('dialTarget').style.transform = 
                `translate(-50%, -100%) rotate(${continuousTargetAngle}deg)`;
            document.getElementById('dialCurrent').textContent = currentAngle.toFixed(1) + '°';
            document.getElementById('dialTargetText').textContent = targetAngle.toFixed(1) + '°';
        }

        function updateTraining(data) {
            document.getElementById('currentEpisode').textContent = data.episode || 0;
            document.getElementById('bestReward').textContent = (data.best_reward || 0).toFixed(2);
            
            if (data.current_params) {
                document.getElementById('trainingStatus').innerHTML = `
                    <div>Episode ${data.episode}/${data.total_episodes}</div>
                    <div style="font-size: 0.9rem; margin-top: 5px;">
                        Testing: Kp=${data.current_params.kp.toFixed(2)}, 
                        Ki=${data.current_params.ki.toFixed(2)}, 
                        Kd=${data.current_params.kd.toFixed(2)}
                    </div>
                    <div style="font-size: 0.8rem; margin-top: 5px; color: #94a3b8;">
                        Current Reward: ${data.current_reward ? data.current_reward.toFixed(2) : '--'}
                    </div>
                `;
            }
            
            if (data.best_params) {
                logMessage('Best so far: Kp=' + data.best_params.kp.toFixed(3) + ', Ki=' + data.best_params.ki.toFixed(3) + ', Kd=' + data.best_params.kd.toFixed(3), 'response');
            }
        }

        function logMessage(message, type = 'info') {
            const log = document.getElementById('logContent');
            const timestamp = new Date().toLocaleTimeString();
            const colors = {
                'command': '#60a5fa', 'response': '#22c55e', 
                'error': '#ef4444', 'warning': '#f59e0b'
            };
            const color = colors[type] || '#cbd5e1';
            
            log.innerHTML += '<div style="color: ' + color + ';">[' + timestamp + '] ' + message + '</div>';
            log.scrollTop = log.scrollHeight;
        }

        window.onload = function() {
            initChart();
            logMessage('Dashboard ready');
        };
    </script>
</body>
</html>'''
    
    with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(html_content)

@app.after_request
def add_header(response):
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response

@app.route('/')
def index():
    """Serve the dashboard"""
    try:
        with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        create_dashboard_template()
        with open('templates/dashboard.html', 'r', encoding='utf-8') as f:
            return f.read()

@socketio.on('connect')
def handle_connect():
    emit('log_message', {'message': 'Web client connected', 'type': 'response'})
    
@socketio.on('ping_test')
def handle_ping_test():
    print("Ping received from web client")
    emit('pong_test', {'message': 'Pong from server!', 'timestamp': time.time()})

@socketio.on('request_status')
def handle_request_status():
    global arduino_controller
    
    print("Manual status request from web client")
    
    if arduino_controller and arduino_controller.connected:
        status = arduino_controller.get_status()
        if status:
            status['connected'] = True
            print(f"Sending manual status: {status}")
            emit('arduino_status', status)
        else:
            emit('arduino_status', {'connected': False})
    else:
        emit('arduino_status', {'connected': False})

@socketio.on('test_websocket')
def handle_test_websocket():
    print("WebSocket test triggered from web interface")
    
    test_data = {
        'connected': True,
        'degrees': 123.4,
        'target_degrees': 90.0,
        'pid_enabled': True,
        'ticks': 1234,
        'error': -33.4,
        'kp': 10.0,
        'ki': 0.0,
        'kd': 30.0
    }
    
    print(f"Sending test data: {test_data}")
    emit('arduino_status', test_data)
    emit('log_message', {'message': 'WebSocket test - if you see this, WebSocket is working!', 'type': 'response'})

@socketio.on('force_status_update')
def handle_force_status():
    global arduino_controller
    
    if arduino_controller and arduino_controller.connected:
        print("Forcing status update...")
        status = arduino_controller.get_status()
        
        if status:
            status['connected'] = True
            print(f"Force sending: {status}")
            emit('arduino_status', status)
        else:
            print("No status data received")
            emit('arduino_status', {'connected': False})
    else:
        print("Arduino not connected")
        emit('arduino_status', {'connected': False})

@socketio.on('connect_arduino')
def handle_arduino_connect(data):
    global arduino_controller, status_monitoring_active
    
    arduino_ip = data.get('ip', '192.168.0.224')
    
    try:
        status_monitoring_active = False
        time.sleep(0.5)
        
        if arduino_controller:
            arduino_controller.disconnect()
        
        arduino_controller = ArduinoPIDController(arduino_ip)
        
        if arduino_controller.connect():
            emit('log_message', {'message': f'Connected to Arduino at {arduino_ip}', 'type': 'response'})
            start_status_monitoring()
        else:
            emit('log_message', {'message': f'Failed to connect to Arduino at {arduino_ip}', 'type': 'error'})
            
    except Exception as e:
        emit('log_message', {'message': f'Connection error: {str(e)}', 'type': 'error'})

@socketio.on('send_command')
def handle_send_command(data):
    global arduino_controller
    
    if not arduino_controller or not arduino_controller.connected:
        emit('log_message', {'message': 'Error: Not connected to Arduino', 'type': 'error'})
        return
    
    command = data.get('command', '').strip()
    emit('log_message', {'message': f'→ {command}', 'type': 'command'})
    
    try:
        response = arduino_controller.send_command(command)
        emit('log_message', {'message': f'← {response}', 'type': 'response'})
    except Exception as e:
        emit('log_message', {'message': f'Command failed: {str(e)}', 'type': 'error'})

@socketio.on('start_rl_training')
def handle_start_training(data):
    global training_active, ppo_optimizer
    
    if not arduino_controller or not arduino_controller.connected:
        emit('log_message', {'message': 'Error: Arduino not connected', 'type': 'error'})
        return
    
    episodes = data.get('episodes', 20)
    duration = data.get('duration', 8.0)
    target = data.get('target', 180.0)
    baseline_params = data.get('baseline_params', None)  # Get baseline from webapp
    
    # Get learning parameters from webapp
    learning_rate = data.get('learning_rate', 0.05)
    exploration_rate = data.get('exploration_rate', 0.3)
    conservative_mode = data.get('conservative_mode', True)
    
    # Initialize PPO optimizer with baseline parameters and learning settings
    ppo_optimizer = PPOPIDOptimizer(
        baseline_params=baseline_params,
        learning_rate=learning_rate,
        exploration_rate=exploration_rate,
        conservative_mode=conservative_mode
    )
    
    training_active = True
    emit('log_message', {'message': f'Starting PPO RL training: {episodes} episodes with baseline integration', 'type': 'command'})
    emit('log_message', {'message': f'Baseline params: {ppo_optimizer.current_params}', 'type': 'info'})
    emit('log_message', {'message': f'Learning settings: Rate={learning_rate}, Exploration={exploration_rate}, Conservative={conservative_mode}', 'type': 'info'})
    
    training_thread = threading.Thread(target=run_ppo_training, args=(episodes, duration, target), daemon=True)
    training_thread.start()

@socketio.on('stop_rl_training')
def handle_stop_training():
    global training_active
    training_active = False
    emit('log_message', {'message': 'Training stopped', 'type': 'warning'})

@socketio.on('update_reward_config')
def handle_update_reward_config(data):
    global reward_config
    
    # Update reward configuration with validation
    for key, value in data.items():
        if key in reward_config:
            if 'weight' in key:
                reward_config[key] = max(0, min(100, float(value)))
            elif key in ['overshoot_penalty', 'stability_sensitivity']:
                reward_config[key] = max(0, min(10, float(value)))
            elif key in ['aggressive_penalty', 'conservative_bonus']:
                reward_config[key] = max(0, min(1, float(value)))
            elif key == 'settling_threshold':
                reward_config[key] = max(0.1, min(10, float(value)))
            elif key == 'settling_window':
                reward_config[key] = max(0.1, min(5, float(value)))
            else:
                reward_config[key] = float(value)
    
    emit('log_message', {'message': 'Reward function configuration updated with settling time parameters', 'type': 'info'})
    emit('reward_config_updated', reward_config)

@socketio.on('save_training_data')
def handle_save_training_data(data):
    global data_logger
    
    save_format = data.get('format', 'both')  # 'json', 'csv', or 'both'
    
    try:
        # Save session data
        files_created = data_logger.save_session(save_format)
        
        # Create visualizations
        plots_created = data_logger.create_visualizations()
        
        emit('training_data_saved', {
            'session_id': data_logger.session_id,
            'files_created': files_created,
            'plots_created': plots_created,
            'message': f'Training data saved successfully! Session: {data_logger.session_id}'
        })
        
        emit('log_message', {
            'message': f'Data saved: {len(files_created)} files, {len(plots_created)} plots created',
            'type': 'success'
        })
        
    except Exception as e:
        emit('log_message', {
            'message': f'Error saving training data: {str(e)}',
            'type': 'error'
        })

@socketio.on('get_reward_config')
def handle_get_reward_config():
    global reward_config
    emit('reward_config', reward_config)

def start_status_monitoring():
    """Start status monitoring thread"""
    global status_monitoring_active
    
    status_monitoring_active = True
    print("Starting status monitoring thread...")
    
    def monitor():
        global status_monitoring_active
        consecutive_failures = 0
        max_failures = 5
        
        while status_monitoring_active and arduino_controller:
            try:
                if arduino_controller.connected:
                    status = arduino_controller.get_status()
                    
                    if status and len(status) > 1:
                        status['connected'] = True
                        
                        # Log data for analysis
                        data_logger.log_status(status)
                        
                        # Rate-limited logging (every 2 seconds)
                        current_time = time.time()
                        if current_time - arduino_controller.last_status_log > 2.0:
                            print(f"Status update: degrees={status.get('degrees', 'N/A')}, target={status.get('target_degrees', 'N/A')}, pid={status.get('pid_enabled', 'N/A')}")
                            arduino_controller.last_status_log = current_time
                        
                        try:
                            socketio.emit('arduino_status', status)
                            # Only log websocket emissions occasionally
                            if current_time - arduino_controller.last_status_log > 2.0:
                                print("Status sent to clients")
                        except Exception as emit_error:
                            print(f"Failed to emit status: {emit_error}")
                        
                        consecutive_failures = 0
                        
                    else:
                        consecutive_failures += 1
                        print(f"Empty status response ({consecutive_failures}/{max_failures})")
                        
                        if consecutive_failures >= max_failures:
                            print("Too many status failures, stopping monitoring")
                            status_monitoring_active = False
                            socketio.emit('arduino_status', {'connected': False})
                            break
                else:
                    print("Arduino not connected, attempting auto-reconnect...")
                    if arduino_controller.auto_reconnect():
                        print("Auto-reconnect successful!")
                        socketio.emit('arduino_status', {'connected': True})
                        continue
                    else:
                        print("Auto-reconnect failed, stopping monitoring")
                        socketio.emit('arduino_status', {'connected': False})
                        break
                    
            except Exception as e:
                consecutive_failures += 1
                print(f"Status monitoring error ({consecutive_failures}/{max_failures}): {e}")
                
                if consecutive_failures >= max_failures:
                    status_monitoring_active = False
                    socketio.emit('arduino_status', {'connected': False})
                    break
            
            time.sleep(0.2)  # Update 5 times per second
        
        print("Status monitoring thread stopped")
        status_monitoring_active = False
    
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

def run_ppo_training(episodes, duration, target):
    """Run PPO-based RL training with safety constraints"""
    global training_active, arduino_controller, ppo_optimizer, data_logger
    
    # Start logging session
    data_logger.start_session("training")
    
    socketio.emit('log_message', {
        'message': 'PPO training initialized with safety constraints and randomized targets',
        'type': 'info'
    })
    
    # Import random for target randomization
    import random
    
    for episode in range(episodes):
        if not training_active:
            break
        
        # System health check before each episode
        if not arduino_controller or not arduino_controller.connected:
            socketio.emit('log_message', {
                'message': f'Training stopped: Arduino disconnected at episode {episode+1}',
                'type': 'error'
            })
            break
        
        # Ensure system is in safe state before test
        initial_status = arduino_controller.get_status()
        if initial_status.get('pid_enabled', False):
            socketio.emit('log_message', {
                'message': f'Episode {episode+1}: PID was still enabled, turning off for safety...',
                'type': 'warning'
            })
            arduino_controller.send_command('PID_OFF')
            time.sleep(0.5)
        
        # Get next parameter set from PPO policy
        test_params = ppo_optimizer.get_next_params(episode)
        
        socketio.emit('log_message', {
            'message': f'Episode {episode+1}: Testing Kp={test_params["kp"]:.3f}, Ki={test_params["ki"]:.3f}, Kd={test_params["kd"]:.3f}',
            'type': 'info'
        })
        
        # Add delay between episodes to let system settle
        if episode > 0:
            socketio.emit('log_message', {
                'message': f'Waiting 2 seconds for system to settle before episode {episode+1}...',
                'type': 'info'
            })
            time.sleep(2.0)
        
        # Randomize target for each episode (vary ±45° from original target)
        episode_target = target + random.uniform(-45, 45)
        episode_target = max(0, min(360, episode_target))  # Keep within 0-360° range
        
        # Test parameters and get reward
        socketio.emit('log_message', {
            'message': f'Starting parameter test for episode {episode+1} - Target: {episode_target:.1f}°...',
            'type': 'info'
        })
        reward = test_parameters_safe(test_params['kp'], test_params['ki'], test_params['kd'], episode_target, duration)
        
        socketio.emit('log_message', {
            'message': f'Episode {episode+1} completed with reward: {reward:.2f}',
            'type': 'response' if reward > 0 else 'warning'
        })
        
        # Log training episode data
        data_logger.log_training_episode(episode + 1, test_params, reward, episode_target, duration)
        
        # Update PPO policy with results
        ppo_optimizer.update_policy(test_params, reward, episode + 1)
        
        # Get training statistics
        stats = ppo_optimizer.get_training_stats()
        
        # Send detailed update to webapp
        socketio.emit('training_update', {
            'episode': episode + 1,
            'total_episodes': episodes,
            'current_params': test_params,
            'current_reward': reward,
            'best_reward': stats['best_reward'],
            'best_params': stats['best_params'],
            'avg_reward_last_10': stats['avg_reward_last_10'],
            'param_stability': stats['param_stability'],
            'exploration_rate': stats['exploration_rate'],
            'safety_status': 'SAFE' if reward > -50 else 'CAUTION'
        })
        
        # Log progress every 5 episodes
        if (episode + 1) % 5 == 0:
            socketio.emit('log_message', {
                'message': f'Episode {episode+1}/{episodes}: Best reward {stats["best_reward"]:.2f}, Avg last 10: {stats["avg_reward_last_10"]:.2f}',
                'type': 'response'
            })
        
        # Brief pause before next episode
        if episode < episodes - 1 and training_active:  # Don't sleep after last episode
            time.sleep(1.0)
    
    # Training complete - ensure system is safe
    training_active = False
    
    # Final safety check - make sure PID is off and motor stopped
    try:
        arduino_controller.send_command('PID_OFF')
        time.sleep(0.2)
        arduino_controller.send_command('STOP')
        socketio.emit('log_message', {
            'message': 'Training complete - system returned to safe state',
            'type': 'info'
        })
    except Exception as e:
        logger.error(f"Error during training cleanup: {e}")
    
    final_stats = ppo_optimizer.get_training_stats()
    
    # Detailed training summary
    socketio.emit('log_message', {
        'message': f'=== TRAINING SUMMARY ===',
        'type': 'info'
    })
    socketio.emit('log_message', {
        'message': f'Episodes completed: {episodes}',
        'type': 'info'
    })
    socketio.emit('log_message', {
        'message': f'Best reward achieved: {final_stats["best_reward"]:.2f}',
        'type': 'response'
    })
    socketio.emit('log_message', {
        'message': f'Best parameters: Kp={final_stats["best_params"]["kp"]:.3f}, Ki={final_stats["best_params"]["ki"]:.3f}, Kd={final_stats["best_params"]["kd"]:.3f}',
        'type': 'response'
    })
    socketio.emit('log_message', {
        'message': f'Parameter stability: {final_stats["param_stability"]:.3f}',
        'type': 'info'
    })
    socketio.emit('log_message', {
        'message': f'Average reward (last 10): {final_stats["avg_reward_last_10"]:.2f}',
        'type': 'info'
    })
    
    socketio.emit('training_complete', {
        'episodes_completed': episodes,
        'best_reward': final_stats['best_reward'],
        'best_params': final_stats['best_params'],
        'final_stability': final_stats['param_stability'],
        'training_method': 'PPO',
        'session_id': data_logger.session_id,
        'ask_save': True
    })
    
    socketio.emit('log_message', {
        'message': f'PPO training complete! Best: Kp={final_stats["best_params"]["kp"]:.3f}, Ki={final_stats["best_params"]["ki"]:.3f}, Kd={final_stats["best_params"]["kd"]:.3f} (Reward: {final_stats["best_reward"]:.2f})',
        'type': 'response'
    })

def test_parameters_safe(kp, ki, kd, target, duration):
    """Test PID parameters with enhanced safety and detailed reward calculation"""
    try:
        # Safety check: ensure parameters are within bounds
        kp = max(0.1, min(50.0, kp))
        ki = max(0.0, min(10.0, ki))
        kd = max(0.0, min(20.0, kd))
        
        # Log what we're about to test
        logger.info(f"Testing parameters: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}, Target={target}°")
        
        # Reset system safely - CRITICAL: Proper sequence
        logger.info("Resetting system for parameter test...")
        response1 = arduino_controller.send_command('PID_OFF')
        logger.info(f"PID_OFF response: {response1}")
        time.sleep(0.5)  # Give time for PID to turn off
        
        response2 = arduino_controller.send_command('STOP')
        logger.info(f"STOP response: {response2}")
        time.sleep(0.3)
        
        response3 = arduino_controller.send_command('ZERO')
        logger.info(f"ZERO response: {response3}")
        time.sleep(1.0)  # Give more time for zeroing
        
        # CRITICAL: Verify system is ready
        initial_status = arduino_controller.get_status()
        if initial_status.get('pid_enabled', True):  # Should be False after PID_OFF
            logger.warning("PID still enabled after PID_OFF command!")
            arduino_controller.send_command('PID_OFF')  # Try again
            time.sleep(0.5)
        
        # Apply parameters with verification
        pid_command = f'PID {kp:.3f} {ki:.3f} {kd:.3f}'
        logger.info(f"Sending PID command: {pid_command}")
        response = arduino_controller.send_command(pid_command)
        logger.info(f"PID command response: {response}")
        
        if response.startswith('ERROR'):
            logger.error(f"PID command failed: {response}")
            return -100  # Severe penalty for failed commands
        
        time.sleep(0.2)  # Let parameters settle
        
        # Verify parameters were applied by checking status
        param_status = arduino_controller.get_status()
        actual_kp = param_status.get('kp', 'unknown')
        actual_ki = param_status.get('ki', 'unknown')
        actual_kd = param_status.get('kd', 'unknown')
        logger.info(f"Arduino reports: Kp={actual_kp}, Ki={actual_ki}, Kd={actual_kd}")
        
        # Set target and enable PID
        target_command = f'TARGET {target}'
        logger.info(f"Sending target command: {target_command}")
        response4 = arduino_controller.send_command(target_command)
        logger.info(f"TARGET response: {response4}")
        time.sleep(0.2)
        
        logger.info("Enabling PID for test...")
        response5 = arduino_controller.send_command('PID_ON')
        logger.info(f"PID_ON response: {response5}")
        
        # Give PID a moment to start
        time.sleep(0.3)
        
        # Verify PID is actually enabled - be more tolerant of empty status
        verify_status = arduino_controller.get_status()
        logger.info(f"Post-enable verification: PID enabled = {verify_status.get('pid_enabled', 'unknown')}, connected = {verify_status.get('connected', 'unknown')}")
        
        # Only fail if we get consistent empty responses (real connection loss)
        if not verify_status or len(verify_status) == 0:
            # Try again once before failing
            time.sleep(0.2)
            verify_status_retry = arduino_controller.get_status()
            if not verify_status_retry or len(verify_status_retry) == 0:
                logger.error("DEBUG: RETURNING -200 - Connection lost during PID enable verification (empty status after retry)!")
                logger.error(f"DEBUG: verify_status = {verify_status}, retry = {verify_status_retry}")
                return -200
            else:
                logger.info("First verification failed but retry succeeded - continuing test")
                verify_status = verify_status_retry
        if not verify_status.get('pid_enabled', False):
            logger.warning("PID reports as disabled, but continuing test (might be Arduino reporting delay)")
            # Don't abort - continue with test as Arduino might have delay in reporting
        
        # Data collection with enhanced metrics and logging
        logger.info(f"Starting data collection for {duration} seconds...")
        start_time = time.time()
        errors = []
        positions = []
        derivatives = []
        data_points = 0
        consecutive_failures = 0  # Track consecutive failed readings
        max_consecutive_failures = 30  # Allow up to 3 seconds of consecutive failures
        
        while time.time() - start_time < duration:
            status = arduino_controller.get_status()
            
            # DEBUG: Log what we're actually getting from Arduino
            logger.info(f"DEBUG: status received = {status}, type = {type(status)}")
            
            # More tolerant data collection - handle delayed or missing data gracefully
            if status and len(status) > 0:
                consecutive_failures = 0  # Reset failure counter on successful read
                
                error = status.get('error', None)
                position = status.get('degrees', None)
                pid_enabled = status.get('pid_enabled', True)  # Default to enabled
                connected = status.get('connected', True)  # Default to connected
                
                # Handle missing data more gracefully
                if error is None or position is None:
                    logger.debug(f"Partial data in status: error={error}, position={position}")
                    # Try to calculate error from position and target if possible
                    if position is not None:
                        error = abs(position - target)
                    elif len(positions) > 0:
                        # Use last known position if current is missing
                        position = positions[-1]
                        error = abs(position - target)
                    else:
                        # Skip this reading rather than use defaults
                        consecutive_failures += 1
                        time.sleep(0.1)
                        continue
                
                errors.append(abs(error))
                positions.append(position)
                data_points += 1
                
                # Log every 20 data points to monitor progress
                if data_points % 20 == 0:
                    logger.info(f"Test progress: {data_points} points, current pos={position:.1f}°, error={error:.1f}°, PID={pid_enabled}")
                
                # Only re-enable PID if we're sure it should be on
                if not pid_enabled and data_points > 10:  # Give some time for PID to activate
                    logger.warning("PID disabled during test! Re-enabling...")
                    arduino_controller.send_command('PID_ON')
                    time.sleep(0.1)
                
                # Calculate derivative for stability assessment
                if len(positions) >= 2:
                    derivative = abs(positions[-1] - positions[-2])
                    derivatives.append(derivative)
            else:
                # Handle missing status data - only abort after sustained failure
                consecutive_failures += 1
                logger.debug(f"No status data received (failure #{consecutive_failures})")
                
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(f"DEBUG: RETURNING -200 due to {consecutive_failures} consecutive failures in data collection loop")
                    arduino_controller.send_command('PID_OFF')
                    return -200
                
                # For occasional missing data, just skip this reading
                # This allows for network delays, Arduino processing delays, etc.
            
            time.sleep(0.1)
        
        # Stop PID safely with logging
        logger.info(f"Test complete. Collected {len(errors)} data points.")
        if positions:
            final_pos = positions[-1]
            final_error = errors[-1] if errors else 0
            logger.info(f"Final position: {final_pos:.1f}°, final error: {final_error:.1f}°")
        
        stop_response1 = arduino_controller.send_command('PID_OFF')
        logger.info(f"Test end PID_OFF response: {stop_response1}")
        time.sleep(0.2)
        stop_response2 = arduino_controller.send_command('STOP')
        logger.info(f"Test end STOP response: {stop_response2}")
        time.sleep(0.3)  # Let motor stop completely
        
        # Calculate comprehensive reward with detailed logging
        expected_data_points = int(duration * 10)  # 10 Hz sampling
        actual_data_points = len(errors)
        
        logger.info(f"Reward calculation: Expected {expected_data_points} points, got {actual_data_points}")
        
        if errors and actual_data_points >= max(5, expected_data_points * 0.3):  # At least 30% of expected data or 5 points minimum
            # Track raw error (signed) to detect overshoot
            signed_errors = []
            for i, position in enumerate(positions):
                signed_error = target - position  # Positive = undershoot, Negative = overshoot
                signed_errors.append(signed_error)
            
            # Use configurable reward parameters
            reward_breakdown = calculate_configurable_reward(
                signed_errors, errors, derivatives, positions, target, kp, ki, kd
            )
            
            return reward_breakdown['total_reward']
        else:
            # Insufficient data collected
            logger.warning(f"Insufficient data for reward calculation: {actual_data_points}/{expected_data_points}")
            return -50
        
    except Exception as e:
        logger.error(f"Parameter test error: {e}")
        # Ensure system is safe after error
        try:
            arduino_controller.send_command('PID_OFF')
            arduino_controller.send_command('STOP')
        except:
            pass
        return -100  # Penalty for test failure

def calculate_configurable_reward(signed_errors, errors, derivatives, positions, target, kp, ki, kd):
    """Configurable reward function based on user preferences"""
    global reward_config
    
    reward_breakdown = {
        'stability_reward': 0,
        'accuracy_reward': 0,
        'cumulative_reward': 0,
        'settling_reward': 0,
        'settling_time_reward': 0,
        'overshoot_penalty': 0,
        'parameter_penalty': 0,
        'conservative_bonus': 0,
        'total_reward': 0
    }
    
    # OVERSHOOT DETECTION AND CONFIGURABLE PENALTY
    max_overshoot = 0
    overshoot_count = 0
    
    for signed_error in signed_errors:
        if signed_error < 0:  # Overshoot detected
            overshoot_amount = abs(signed_error)
            max_overshoot = max(max_overshoot, overshoot_amount)
            overshoot_count += 1
    
    if overshoot_count > 0:
        overshoot_percentage = overshoot_count / len(signed_errors) * 100
        overshoot_penalty = (
            max_overshoot * reward_config['overshoot_penalty'] +
            overshoot_percentage * (reward_config['overshoot_penalty'] / 2) +
            (100 if max_overshoot > 10 else 0)
        )
        reward_breakdown['overshoot_penalty'] = overshoot_penalty
        logger.info(f"OVERSHOOT: Max={max_overshoot:.1f}°, {overshoot_percentage:.1f}% of time, penalty={overshoot_penalty:.1f}")
    
    # STABILITY COMPONENT (configurable)
    stability_reward = 100
    if derivatives and len(derivatives) > 5:
        mean_derivative = np.mean(derivatives)
        max_derivative = np.max(derivatives) if derivatives else 0
        derivative_std = np.std(derivatives) if len(derivatives) > 1 else 0
        
        stability_penalty = (
            mean_derivative * reward_config['stability_sensitivity'] +
            max_derivative * (reward_config['stability_sensitivity'] / 2) +
            derivative_std * reward_config['stability_sensitivity']
        )
        stability_reward = max(0, stability_reward - stability_penalty)
        
        logger.info(f"Stability: mean_deriv={mean_derivative:.2f}, max_deriv={max_derivative:.2f}, std={derivative_std:.2f}")
    
    reward_breakdown['stability_reward'] = stability_reward
    
    # ACCURACY AND CUMULATIVE ERROR
    cumulative_error = np.sum(errors)
    mean_error = np.mean(errors)
    
    cumulative_reward = max(0, 50 - cumulative_error/10)
    accuracy_reward = max(0, 50 - mean_error)
    
    reward_breakdown['cumulative_reward'] = cumulative_reward
    reward_breakdown['accuracy_reward'] = accuracy_reward
    
    # SETTLING PERFORMANCE
    final_portion = 0.3
    final_errors = errors[-int(len(errors)*final_portion):] if len(errors) > 10 else errors
    final_signed_errors = signed_errors[-int(len(signed_errors)*final_portion):] if len(signed_errors) > 10 else signed_errors
    
    settling_reward = 50
    if final_errors:
        final_mean_error = np.mean(final_errors)
        final_std_error = np.std(final_errors) if len(final_errors) > 1 else 0
        final_max_error = np.max(final_errors)
        final_overshoot = any(err < 0 for err in final_signed_errors)
        
        settling_penalty = (
            final_mean_error * 2.0 +
            final_std_error * 5.0 +
            final_max_error * 1.5 +
            (50 if final_overshoot else 0)
        )
        settling_reward = max(0, settling_reward - settling_penalty)
    
    reward_breakdown['settling_reward'] = settling_reward
    
    # SETTLING TIME COMPONENT (you value fast settling)
    settling_time_reward = 50  # Start with base reward
    settling_threshold = reward_config['settling_threshold']
    settling_window_seconds = reward_config['settling_window']
    
    # Calculate settling time - time to reach and maintain stable position
    settling_time = None
    data_rate = 10  # 10 Hz sampling rate
    window_size = max(1, int(settling_window_seconds * data_rate))
    
    if len(errors) > window_size:
        # Look for the first point where error stays within threshold for the required window
        for i in range(window_size, len(errors)):
            window_errors = errors[i-window_size:i]
            if all(err <= settling_threshold for err in window_errors):
                # Also check for no overshoot in this window
                window_signed_errors = signed_errors[i-window_size:i] if i <= len(signed_errors) else []
                if not any(err < 0 for err in window_signed_errors):  # No overshoot
                    settling_time = (i - window_size) / data_rate  # Time in seconds
                    break
        
        if settling_time is not None:
            # Reward faster settling (lower settling time = higher reward)
            max_settling_time = len(errors) / data_rate  # Total test duration
            settling_time_ratio = settling_time / max_settling_time
            
            # Linear reward: 100% for instant settling, 0% for never settling
            settling_time_reward = max(0, 50 * (1.0 - settling_time_ratio))
            
            logger.info(f"Settling time: {settling_time:.2f}s (ratio: {settling_time_ratio:.2f})")
        else:
            # Never settled - heavy penalty
            settling_time_reward = 0
            logger.info("System never settled within threshold")
    
    reward_breakdown['settling_time_reward'] = settling_time_reward
    
    # COMBINE REWARDS WITH CONFIGURABLE WEIGHTS (normalize to 100)
    weight_sum = (reward_config['stability_weight'] + reward_config['accuracy_weight'] + 
                  reward_config['cumulative_weight'] + reward_config['settling_weight'] +
                  reward_config['settling_time_weight'])
    
    if weight_sum > 0:
        base_reward = (
            (reward_config['stability_weight'] / weight_sum) * stability_reward +
            (reward_config['accuracy_weight'] / weight_sum) * accuracy_reward +
            (reward_config['cumulative_weight'] / weight_sum) * cumulative_reward +
            (reward_config['settling_weight'] / weight_sum) * settling_reward +
            (reward_config['settling_time_weight'] / weight_sum) * settling_time_reward
        )
    else:
        base_reward = (stability_reward + accuracy_reward + cumulative_reward + settling_reward + settling_time_reward) / 5
    
    # Apply penalties and bonuses
    total_reward = base_reward - reward_breakdown['overshoot_penalty']
    
    # Parameter aggressiveness penalty
    if kp > 20 or ki > 3 or kd > 10:
        penalty_factor = 1.0 - reward_config['aggressive_penalty']
        total_reward *= penalty_factor
        reward_breakdown['parameter_penalty'] = base_reward * reward_config['aggressive_penalty']
    
    # Conservative behavior bonus
    if max_overshoot == 0 and stability_reward > 80:
        bonus_factor = 1.0 + reward_config['conservative_bonus']
        total_reward *= bonus_factor
        reward_breakdown['conservative_bonus'] = total_reward * reward_config['conservative_bonus'] / bonus_factor
    
    reward_breakdown['total_reward'] = total_reward
    
    logger.info(f"REWARD BREAKDOWN: stability={stability_reward:.1f}, accuracy={accuracy_reward:.1f}, "
                f"cumulative={cumulative_reward:.1f}, settling={settling_reward:.1f}, "
                f"settling_time={settling_time_reward:.1f}, total={total_reward:.1f}")
    
    return reward_breakdown

if __name__ == '__main__':
    print("=" * 60)
    print("PID CONTROLLER WEB SERVER")
    print("=" * 60)
    
    local_ip = get_local_ip()
    
    print("Loading dashboard template from templates/dashboard.html")
    print()
    print(f"Starting server...")
    print(f"   Local:    http://localhost:5000")
    print(f"   Network:  http://{local_ip}:5000")
    print()
    print("System features:")
    print("   • PID motor control")
    print("   • Real-time position tracking") 
    print("   • Manual motor controls")
    print("   • Reinforcement learning training")
    print("   • Connection stability improved")
    print()
    print("Required: pip install flask flask-socketio numpy")
    print("=" * 60)
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nServer stopped")
        if arduino_controller:
            arduino_controller.disconnect()