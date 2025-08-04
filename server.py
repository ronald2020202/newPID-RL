#!/usr/bin/env python3
"""
Complete Flask Server with Online PPO Training
Fast, intelligent PID parameter optimization using PyTorch
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
import os

# PyTorch imports with fallback
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
    PYTORCH_AVAILABLE = True
    print("✅ PyTorch available - Online PPO training enabled")
except ImportError:
    PYTORCH_AVAILABLE = False
    print("❌ PyTorch not available - install with: pip install torch")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArduinoPIDController:
    def __init__(self, arduino_ip: str, arduino_port: int = 80, timeout: float = 3.0):
        self.arduino_ip = arduino_ip
        self.arduino_port = arduino_port
        self.timeout = timeout
        self.socket = None
        self.connected = False
        self.status_data = {}
        self.last_heartbeat = 0
        
    def connect(self) -> bool:
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
            logger.info(f"Arduino: {response}")
            
            self.connected = True
            self.last_heartbeat = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        self.connected = False
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
    
    def send_command(self, command: str) -> str:
        if not self.connected:
            return "ERROR Not connected"
        
        try:
            self.socket.send((command + '\n').encode())
            response = self._read_response()
            self.last_heartbeat = time.time()
            return response
        except Exception as e:
            logger.error(f"Command error: {e}")
            self.connected = False
            return f"ERROR {e}"
    
    def _read_response(self) -> str:
        try:
            data = self.socket.recv(1024).decode('utf-8', errors='ignore').strip()
            return data
        except Exception as e:
            return f"ERROR {e}"
    
    def get_status(self):
        try:
            response = self.send_command("STATUS")
            if response.startswith("STATUS "):
                return self._parse_status(response[7:])
            return {}
        except:
            return {}
    
    def _parse_status(self, status_string: str):
        try:
            pairs = status_string.split(',')
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
            logger.error(f"Status parse error: {e}")
            return {}

# PPO Agent Class (only if PyTorch available)
if PYTORCH_AVAILABLE:
    class PPOAgent:
        def __init__(self, state_dim=5, action_dim=3, lr=0.0005):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Small networks for fast online learning
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, action_dim * 2)  # mean and std for each action
            ).to(self.device)
            
            self.critic = nn.Sequential(
                nn.Linear(state_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1)
            ).to(self.device)
            
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()), 
                lr=lr
            )
            
            # PPO hyperparameters
            self.clip_epsilon = 0.15
            self.gamma = 0.9
            
            # Initialize around your known working parameters
            self.baseline_params = np.array([10.0, 0.0, 30.0])  # Your Kp, Ki, Kd
            self.param_scales = np.array([3.0, 0.5, 10.0])     # Search ranges
            
            # Experience buffer
            self.buffer = {
                'states': [], 'actions': [], 'rewards': [], 
                'values': [], 'log_probs': []
            }
        
        def get_action(self, state):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.actor(state_tensor)
                means = output[:, :3]
                log_stds = output[:, 3:]
                stds = torch.exp(torch.clamp(log_stds, -1, 1))
                
                dist = Normal(means, stds)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum()
                value = self.critic(state_tensor)
                
                # Convert to PID parameters with small adjustments
                adjustments = action.cpu().numpy().flatten() * 0.15
                pid_params = self.baseline_params + adjustments * self.param_scales
                
                # Clamp to reasonable ranges
                pid_params[0] = np.clip(pid_params[0], 2.0, 20.0)   # Kp
                pid_params[1] = np.clip(pid_params[1], 0.0, 2.0)    # Ki
                pid_params[2] = np.clip(pid_params[2], 10.0, 50.0)  # Kd
                
                return pid_params, log_prob.item(), value.item(), action.cpu().numpy().flatten()
        
        def store_experience(self, state, action, reward, value, log_prob):
            self.buffer['states'].append(state)
            self.buffer['actions'].append(action)
            self.buffer['rewards'].append(reward)
            self.buffer['values'].append(value)
            self.buffer['log_probs'].append(log_prob)
        
        def update_policy(self):
            if len(self.buffer['states']) < 5:
                return
            
            states = torch.FloatTensor(self.buffer['states']).to(self.device)
            actions = torch.FloatTensor(self.buffer['actions']).to(self.device)
            old_log_probs = torch.FloatTensor(self.buffer['log_probs']).to(self.device)
            values = torch.FloatTensor(self.buffer['values']).to(self.device)
            rewards = self.buffer['rewards']
            
            # Calculate returns
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + self.gamma * R
                returns.insert(0, R)
            
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # PPO update
            for _ in range(3):
                output = self.actor(states)
                means = output[:, :3]
                log_stds = output[:, 3:]
                stds = torch.exp(torch.clamp(log_stds, -1, 1))
                
                dist = Normal(means, stds)
                new_log_probs = dist.log_prob(actions).sum(dim=-1)
                new_values = self.critic(states).squeeze()
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                clipped = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                
                actor_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
                critic_loss = nn.MSELoss()(new_values, returns)
                
                loss = actor_loss + 0.5 * critic_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Clear buffer
            for key in self.buffer:
                self.buffer[key] = []

    class OnlineTrainer:
        def __init__(self, controller, socketio):
            self.controller = controller
            self.socketio = socketio
            self.agent = PPOAgent()
            self.training_active = False
            self.episode_count = 0
            self.best_reward = -float('inf')
            self.best_params = None
            self.recent_rewards = []
        
        def start_training(self, max_episodes, target_angle):
            self.training_active = True
            self.target_angle = target_angle
            
            self.socketio.emit('log_message', {
                'message': f'🚀 Starting Online PPO: {max_episodes} episodes, target={target_angle}°',
                'type': 'command'
            })
            
            thread = threading.Thread(target=self._train_loop, args=(max_episodes,), daemon=True)
            thread.start()
        
        def _train_loop(self, max_episodes):
            for episode in range(max_episodes):
                if not self.training_active:
                    break
                
                try:
                    # Get current system state
                    state = self._get_state()
                    
                    # Get action from PPO policy
                    pid_params, log_prob, value, raw_action = self.agent.get_action(state)
                    
                    # Test parameters on real hardware
                    reward = self._quick_test(pid_params)
                    
                    # Store experience for learning
                    self.agent.store_experience(state, raw_action, reward, value, log_prob)
                    
                    # Update tracking
                    self.recent_rewards.append(reward)
                    if len(self.recent_rewards) > 10:
                        self.recent_rewards.pop(0)
                    
                    if reward > self.best_reward:
                        self.best_reward = reward
                        self.best_params = pid_params.copy()
                        
                        self.socketio.emit('log_message', {
                            'message': f'🎯 NEW BEST! Reward: {reward:.2f}, Kp={pid_params[0]:.2f}, Ki={pid_params[1]:.2f}, Kd={pid_params[2]:.2f}',
                            'type': 'response'
                        })
                    
                    # Update policy every 5 episodes
                    if episode % 5 == 0 and episode > 0:
                        self.agent.update_policy()
                        avg_reward = np.mean(self.recent_rewards)
                        
                        self.socketio.emit('log_message', {
                            'message': f'🧠 Policy updated! Avg reward: {avg_reward:.2f}',
                            'type': 'response'
                        })
                    
                    # Send progress update
                    self.socketio.emit('training_update', {
                        'episode': episode + 1,
                        'total_episodes': max_episodes,
                        'current_reward': reward,
                        'best_reward': self.best_reward,
                        'best_params': {
                            'kp': self.best_params[0],
                            'ki': self.best_params[1],
                            'kd': self.best_params[2]
                        } if self.best_params is not None else None,
                        'current_params': {
                            'kp': pid_params[0],
                            'ki': pid_params[1],
                            'kd': pid_params[2]
                        },
                        'avg_recent_reward': np.mean(self.recent_rewards)
                    })
                    
                    time.sleep(0.8)  # Fast episodes
                    
                except Exception as e:
                    self.socketio.emit('log_message', {
                        'message': f'❌ Episode {episode + 1} error: {str(e)}',
                        'type': 'error'
                    })
                    time.sleep(1.0)
            
            # Training complete
            self.training_active = False
            if self.best_params is not None:
                kp, ki, kd = self.best_params
                self.controller.send_command(f'PID {kp:.3f} {ki:.3f} {kd:.3f}')
                
                self.socketio.emit('log_message', {
                    'message': f'🏆 PPO COMPLETE! Applied best: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f} (Reward: {self.best_reward:.2f})',
                    'type': 'response'
                })
        
        def _get_state(self):
            try:
                status = self.controller.get_status()
                return np.array([
                    (status.get('degrees', 0) % 360) / 360.0,
                    (status.get('target_degrees', 0) % 360) / 360.0,
                    np.clip(status.get('error', 0) / 180.0, -1, 1),
                    min(status.get('stable_count', 0) / 50.0, 1.0),
                    float(status.get('pid_enabled', False))
                ], dtype=np.float32)
            except:
                return np.zeros(5, dtype=np.float32)
        
        def _quick_test(self, pid_params):
            kp, ki, kd = pid_params
            
            try:
                # Quick 3-second test
                self.controller.send_command('PID_OFF')
                time.sleep(0.1)
                self.controller.send_command(f'PID {kp:.3f} {ki:.3f} {kd:.3f}')
                self.controller.send_command(f'TARGET {self.target_angle}')
                self.controller.send_command('PID_ON')
                
                # Collect data
                start_time = time.time()
                errors = []
                stable_counts = []
                
                while time.time() - start_time < 3.0:
                    status = self.controller.get_status()
                    errors.append(abs(status.get('error', 0)))
                    stable_counts.append(status.get('stable_count', 0))
                    time.sleep(0.15)
                
                self.controller.send_command('PID_OFF')
                
                # Calculate reward
                if not errors:
                    return -50
                
                mean_error = np.mean(errors)
                max_stable = max(stable_counts) if stable_counts else 0
                final_error = errors[-1]
                
                # Reward function optimized for online learning
                reward = max(0, 120 - mean_error * 3)
                
                if max_stable >= 20:
                    reward += 40
                if max_stable >= 35:
                    reward += 30
                
                reward -= final_error * 2
                
                return reward
                
            except Exception as e:
                logger.error(f"Test error: {e}")
                return -100

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'ppo-pid-controller'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
arduino_controller = None
online_trainer = None

def create_dashboard_template():
    os.makedirs('templates', exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PPO PID Controller</title>
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
            <h1>🧠 PPO PID Controller</h1>
            <p>Online reinforcement learning for optimal control</p>
        </div>

        <!-- Connection -->
        <div class="card">
            <div class="card-title">Connection Status</div>
            <div style="display: flex; align-items: center; gap: 15px; flex-wrap: wrap;">
                <div class="status-indicator disconnected" id="connectionIndicator"></div>
                <span id="connectionText">Disconnected</span>
                <input type="text" id="arduinoIP" placeholder="Arduino IP" value="192.168.1.100" style="width: 150px;">
                <button class="button" onclick="connectArduino()">Connect</button>
                <div id="lastUpdate" style="font-size: 0.8rem; color: #94a3b8;">Last: Never</div>
            </div>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px;">
            <!-- Status -->
            <div class="card">
                <div class="card-title">System Status</div>
                <div class="status-grid">
                    <div class="status-item">
                        <div class="status-value" id="currentDegrees">0.0°</div>
                        <div class="status-label">Current Position</div>
                    </div>
                    <div class="status-item">
                        <div class="status-value" id="targetDegrees">0.0°</div>
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
                    <input type="number" id="targetInput" placeholder="Target degrees" value="90" step="0.1">
                    <button class="button" onclick="setTarget()">Set Target</button>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; text-align: center; margin-top: 10px;">
                    Press ENTER for emergency stop
                </div>
            </div>

            <!-- PID -->
            <div class="card">
                <div class="card-title">PID Control</div>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                    <div class="input-group">
                        <label>Kp</label>
                        <input type="number" id="kpValue" step="0.1" value="10.0">
                    </div>
                    <div class="input-group">
                        <label>Ki</label>
                        <input type="number" id="kiValue" step="0.01" value="0.0">
                    </div>
                    <div class="input-group">
                        <label>Kd</label>
                        <input type="number" id="kdValue" step="0.1" value="30.0">
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
                        Current: <span id="dialCurrent">0.0°</span><br>
                        Target: <span id="dialTargetText">0.0°</span>
                    </div>
                </div>
                <div class="chart-container">
                    <canvas id="positionChart"></canvas>
                </div>
            </div>
        </div>

        <!-- PPO Training -->
        <div class="card" style="background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(16, 185, 129, 0.1) 100%);">
            <div class="card-title">🧠 Online PPO Training</div>
            <div style="display: grid; grid-template-columns: 1fr 2fr 1fr; gap: 20px;">
                <div>
                    <div class="input-group">
                        <label>Episodes</label>
                        <input type="number" id="episodes" value="50" min="10" max="200">
                    </div>
                    <div class="input-group">
                        <label>Target Angle (°)</label>
                        <input type="number" id="rlTarget" value="180" step="0.1">
                    </div>
                    <button class="button success" onclick="startTraining()">🚀 Start PPO</button>
                    <button class="button danger" onclick="stopTraining()">⏹ Stop</button>
                    <button class="button" onclick="updateBaseline()" style="width: 100%; margin-top: 10px; font-size: 0.8rem;">📍 Update Baseline</button>
                </div>
                <div id="trainingStatus" style="text-align: center; color: #94a3b8; padding: 20px;">
                    Ready for PPO training
                </div>
                <div style="text-align: center;">
                    <div>Episode: <span id="currentEpisode">0</span></div>
                    <div>Best Reward: <span id="bestReward">--</span></div>
                    <div style="margin-top: 10px; font-size: 0.9rem;">
                        <div>🎯 Best Kp: <span id="bestKp">--</span></div>
                        <div>🎯 Best Ki: <span id="bestKi">--</span></div>
                        <div>🎯 Best Kd: <span id="bestKd">--</span></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Log -->
        <div class="log-container">
            <div style="color: #60a5fa; margin-bottom: 10px;">🤖 PPO Training Log</div>
            <div id="logContent">Ready to connect and start learning...</div>
        </div>
    </div>

    <script>
        const socket = io();
        let connected = false;
        let positionChart;
        let chartData = [];
        let currentAngle = 0;
        let targetAngle = 0;

        // Emergency stop on Enter
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' || event.key === ' ') {
                event.preventDefault();
                sendCommand('STOP');
                logMessage('🚨 EMERGENCY STOP activated', 'warning');
            }
        });

        // Initialize chart
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
                        y: { min: 0, max: 360, ticks: { color: '#94a3b8' } }
                    },
                    animation: false
                }
            });
        }

        // Socket events
        socket.on('connect', () => {
            logMessage('🌐 Connected to PPO server');
        });

        socket.on('arduino_status', (data) => {
            updateStatus(data);
        });

        socket.on('log_message', (data) => {
            logMessage(data.message, data.type);
        });

        socket.on('training_update', (data) => {
            updateTraining(data);
        });

        // Functions
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
            sendCommand(`PID ${kp} ${ki} ${kd}`);
        }

        function setTarget() {
            const target = document.getElementById('targetInput').value;
            sendCommand(`TARGET ${target}`);
        }

        function startTraining() {
            if (!connected) {
                logMessage('❌ Must be connected to Arduino for PPO training', 'error');
                return;
            }
            
            const episodes = document.getElementById('episodes').value;
            const target = document.getElementById('rlTarget').value;
            
            socket.emit('start_online_training', {
                episodes: parseInt(episodes),
                target: parseFloat(target)
            });
        }

        function stopTraining() {
            socket.emit('stop_online_training');
        }

        function updateBaseline() {
            const kp = parseFloat(document.getElementById('kpValue').value);
            const ki = parseFloat(document.getElementById('kiValue').value);
            const kd = parseFloat(document.getElementById('kdValue').value);
            
            socket.emit('update_baseline', {kp: kp, ki: ki, kd: kd});
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
            
            // Handle angle wrapping for smooth chart
            let displayCurrent = currentAngle;
            let displayTarget = targetAngle;
            
            if (chartData.length > 0) {
                const lastCurrent = chartData[chartData.length - 1].current;
                const lastTarget = chartData[chartData.length - 1].target;
                
                if (Math.abs(currentAngle - lastCurrent) > 180) {
                    displayCurrent = currentAngle + (currentAngle < lastCurrent ? 360 : -360);
                }
                
                if (Math.abs(targetAngle - lastTarget) > 180) {
                    displayTarget = targetAngle + (targetAngle < lastTarget ? 360 : -360);
                }
            }
            
            chartData.push({time: now, current: displayCurrent, target: displayTarget});
            
            if (chartData.length > 50) chartData.shift();
            
            positionChart.data.labels = chartData.map(d => d.time);
            positionChart.data.datasets[0].data = chartData.map(d => d.current);
            positionChart.data.datasets[1].data = chartData.map(d => d.target);
            positionChart.update('none');
        }

        function updateDial() {
            const pointer = document.getElementById('dialPointer');
            const target = document.getElementById('dialTarget');
            
            // Smart rotation for shortest path
            const currentTransform = pointer.style.transform;
            const currentRotation = currentTransform.match(/rotate\\(([^)]+)deg\\)/);
            const lastAngle = currentRotation ? parseFloat(currentRotation[1]) : 0;
            
            let newAngle = currentAngle;
            let angleDiff = newAngle - lastAngle;
            
            while (angleDiff > 180) angleDiff -= 360;
            while (angleDiff < -180) angleDiff += 360;
            
            const finalAngle = lastAngle + angleDiff;
            
            pointer.style.transform = `translate(-50%, -100%) rotate(${finalAngle}deg)`;
            target.style.transform = `translate(-50%, -100%) rotate(${targetAngle}deg)`;
            
            document.getElementById('dialCurrent').textContent = currentAngle.toFixed(1) + '°';
            document.getElementById('dialTargetText').textContent = targetAngle.toFixed(1) + '°';
        }

        function updateTraining(data) {
            document.getElementById('currentEpisode').textContent = data.episode || 0;
            document.getElementById('bestReward').textContent = (data.best_reward || 0).toFixed(2);
            
            if (data.best_params) {
                document.getElementById('bestKp').textContent = data.best_params.kp.toFixed(2);
                document.getElementById('bestKi').textContent = data.best_params.ki.toFixed(2);
                document.getElementById('bestKd').textContent = data.best_params.kd.toFixed(2);
            }
            
            if (data.current_params) {
                document.getElementById('trainingStatus').innerHTML = `
                    <div style="color: #22c55e; font-size: 1.1rem;">🧠 PPO Learning</div>
                    <div style="margin-top: 8px;">Episode ${data.episode}/${data.total_episodes}</div>
                    <div style="font-size: 0.9rem; margin-top: 8px;">
                        Testing: Kp=${data.current_params.kp.toFixed(2)}, 
                        Ki=${data.current_params.ki.toFixed(2)}, 
                        Kd=${data.current_params.kd.toFixed(2)}
                    </div>
                    <div style="font-size: 0.8rem; margin-top: 5px; color: #94a3b8;">
                        Reward: ${data.current_reward ? data.current_reward.toFixed(2) : '--'}
                        ${data.avg_recent_reward ? ' (Avg: ' + data.avg_recent_reward.toFixed(2) + ')' : ''}
                    </div>
                `;
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
            
            log.innerHTML += `<div style="color: ${color};">[${timestamp}] ${message}</div>`;
            log.scrollTop = log.scrollHeight;
        }

        // Initialize
        window.onload = function() {
            initChart();
            logMessage('🚀 PPO PID Controller ready');
        };
    </script>
</body>
</html>'''
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(html_content)

@app.route('/')
def index():
    try:
        with open('templates/dashboard.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        create_dashboard_template()
        with open('templates/dashboard.html', 'r') as f:
            return f.read()

# Socket event handlers
@socketio.on('connect')
def handle_connect():
    emit('log_message', {'message': '🌐 Web client connected', 'type': 'response'})

@socketio.on('connect_arduino')
def handle_arduino_connect(data):
    global arduino_controller
    
    arduino_ip = data.get('ip', '192.168.1.100')
    
    try:
        if arduino_controller:
            arduino_controller.disconnect()
        
        arduino_controller = ArduinoPIDController(arduino_ip)
        
        if arduino_controller.connect():
            emit('log_message', {'message': f'✅ Connected to Arduino at {arduino_ip}', 'type': 'response'})
            start_status_monitoring()
        else:
            emit('log_message', {'message': f'❌ Failed to connect to Arduino at {arduino_ip}', 'type': 'error'})
            
    except Exception as e:
        emit('log_message', {'message': f'❌ Connection error: {str(e)}', 'type': 'error'})

@socketio.on('send_command')
def handle_send_command(data):
    global arduino_controller
    
    if not arduino_controller or not arduino_controller.connected:
        emit('log_message', {'message': '❌ Not connected to Arduino', 'type': 'error'})
        return
    
    command = data.get('command', '').strip()
    emit('log_message', {'message': f'📤 {command}', 'type': 'command'})
    
    try:
        response = arduino_controller.send_command(command)
        emit('log_message', {'message': f'📥 {response}', 'type': 'response'})
    except Exception as e:
        emit('log_message', {'message': f'❌ Command failed: {str(e)}', 'type': 'error'})

# PPO Training handlers (only if PyTorch available)
if PYTORCH_AVAILABLE:
    @socketio.on('start_online_training')
    def handle_start_online_training(data):
        global online_trainer, arduino_controller
        
        if not arduino_controller or not arduino_controller.connected:
            emit('log_message', {'message': '❌ Arduino not connected for PPO training', 'type': 'error'})
            return
        
        episodes = data.get('episodes', 50)
        target = data.get('target', 180.0)
        
        online_trainer = OnlineTrainer(arduino_controller, socketio)
        online_trainer.start_training(episodes, target)

    @socketio.on('stop_online_training')
    def handle_stop_online_training():
        global online_trainer
        if online_trainer:
            online_trainer.training_active = False

    @socketio.on('update_baseline')
    def handle_update_baseline(data):
        global online_trainer
        if online_trainer and online_trainer.agent:
            kp = data.get('kp', 10.0)
            ki = data.get('ki', 0.0)
            kd = data.get('kd', 30.0)
            
            online_trainer.agent.baseline_params = np.array([kp, ki, kd])
            emit('log_message', {'message': f'📍 Baseline updated: Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f}', 'type': 'response'})
else:
    @socketio.on('start_online_training')
    def handle_no_pytorch(data):
        emit('log_message', {'message': '❌ PyTorch required for PPO. Install: pip install torch', 'type': 'error'})

def start_status_monitoring():
    def monitor():
        while arduino_controller and arduino_controller.connected:
            try:
                status = arduino_controller.get_status()
                if status:
                    status['connected'] = True
                    socketio.emit('arduino_status', status)
                else:
                    socketio.emit('arduino_status', {'connected': False})
            except:
                socketio.emit('arduino_status', {'connected': False})
                break
            time.sleep(0.5)
    
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

if __name__ == '__main__':
    print("🧠 PPO PID Controller Server")
    print("=" * 50)
    
    if PYTORCH_AVAILABLE:
        print("✅ PyTorch detected - Full PPO training available")
    else:
        print("❌ PyTorch not found - Please install: pip install torch")
    
    print("📁 Creating dashboard template...")
    create_dashboard_template()
    print("✅ Dashboard created!")
    print()
    print("🌐 Starting server on http://localhost:5000")
    print("📋 Required packages: flask flask-socketio numpy torch")
    print()
    print("🎯 Features:")
    print("   - Online PPO learning (3-second episodes)")
    print("   - Smart parameter exploration around your baseline")
    print("   - Real-time hardware testing")
    print("   - Automatic best parameter application")
    print()
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        if arduino_controller:
            arduino_controller.disconnect()