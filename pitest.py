#!/usr/bin/env python3
"""
Your original Flask backend for PID Controller Dashboard
Fixed to prevent connection loops
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
        self.running = True
        self.last_heartbeat = 0
        
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
            
            # Wait for connection message
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
            self.socket.send((command + '\n').encode())
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
            data = self.socket.recv(1024).decode('utf-8', errors='ignore').strip()
            return data
        except Exception as e:
            return f"ERROR {e}"
    
    def get_status(self) -> Dict:
        """Get current status with better error handling"""
        try:
            response = self.send_command("STATUS")
            print(f"🔍 Raw status response: '{response}'")  # Debug output
            
            if response.startswith("STATUS "):
                status_data = self._parse_status(response[7:])
                print(f"🔍 Parsed status: {status_data}")  # Debug output
                return status_data
            else:
                print(f"⚠️  Unexpected status response: '{response}'")
                return {}
        except Exception as e:
            print(f"❌ Status error: {e}")
            return {}
    
    def _parse_status(self, status_string: str) -> Dict:
        """Parse status string into dictionary with better error handling"""
        try:
            print(f"🔍 Parsing status string: '{status_string}'")
            
            pairs = status_string.split(',')
            status_dict = {}
            
            for pair in pairs:
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Convert values
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
                            pass  # Keep as string
                    
                    status_dict[key] = value
                    print(f"🔍   {key} = {value}")
            
            status_dict['timestamp'] = time.time()
            return status_dict
            
        except Exception as e:
            print(f"❌ Status parsing error: {e}")
            return {}
    
    def is_healthy(self) -> bool:
        """Check if connection is healthy"""
        return self.connected and (time.time() - self.last_heartbeat < 10.0)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = 'pid-controller-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables
arduino_controller = None
training_active = False
training_data = {
    'active': False,
    'episode': 0,
    'total_episodes': 0,
    'best_reward': -float('inf'),
    'best_params': None,
    'completed': False
}
status_monitoring_active = False

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
                    <label>Target Position (degrees)</label>
                    <input type="number" id="targetInput" placeholder="Enter target in degrees" value="90" step="0.1" min="0" max="360">
                    <button class="button" onclick="setTarget()">Set Target</button>
                </div>
                
                <div class="input-group">
                    <label>Quick Targets</label>
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 5px;">
                        <button class="button" onclick="setQuickTarget(0)">0°</button>
                        <button class="button" onclick="setQuickTarget(90)">90°</button>
                        <button class="button" onclick="setQuickTarget(180)">180°</button>
                        <button class="button" onclick="setQuickTarget(270)">270°</button>
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
            logMessage('Connected to server');
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
            const target = parseFloat(document.getElementById('targetInput').value);
            sendCommand(`TARGET ${target}`);
        }

        function setQuickTarget(degrees) {
            sendCommand(`TARGET ${degrees}`);
        }

        // Add keyboard event listener for emergency stop
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
                
                // Update displays
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
            chartData.push({time: now, current: currentAngle, target: targetAngle});
            
            if (chartData.length > 50) chartData.shift();
            
            positionChart.data.labels = chartData.map(d => d.time);
            positionChart.data.datasets[0].data = chartData.map(d => d.current);
            positionChart.data.datasets[1].data = chartData.map(d => d.target);
            positionChart.update('none');
        }

        function updateDial() {
            document.getElementById('dialPointer').style.transform = 
                `translate(-50%, -100%) rotate(${currentAngle}deg)`;
            document.getElementById('dialTarget').style.transform = 
                `translate(-50%, -100%) rotate(${targetAngle}deg)`;
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
                // Update best parameters display (you can add this section to HTML if needed)
                logMessage(`Best so far: Kp=${data.best_params.kp.toFixed(3)}, Ki=${data.best_params.ki.toFixed(3)}, Kd=${data.best_params.kd.toFixed(3)}`, 'response');
            }
        }

        // Add training complete handler
        socket.on('training_complete', (data) => {
            logMessage(`Training complete! Episodes: ${data.episodes_completed}, Best reward: ${data.best_reward.toFixed(2)}`, 'response');
            
            if (data.best_params) {
                // Auto-fill the PID form with best parameters
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
            logMessage('Dashboard ready');
        };
    </script>
</body>
</html>'''
    
    with open('templates/dashboard.html', 'w') as f:
        f.write(html_content)

@app.route('/')
def index():
    """Serve the dashboard"""
    try:
        with open('templates/dashboard.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        create_dashboard_template()
        with open('templates/dashboard.html', 'r') as f:
            return f.read()

# WebSocket handlers
@socketio.on('connect')
def handle_connect():
    emit('log_message', {'message': 'Web client connected', 'type': 'response'})

@socketio.on('connect_arduino')
def handle_arduino_connect(data):
    global arduino_controller, status_monitoring_active
    
    arduino_ip = data.get('ip', '192.168.0.224')
    
    try:
        # Stop any existing monitoring
        status_monitoring_active = False
        time.sleep(0.5)  # Give time for monitoring to stop
        
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
    global training_active
    
    if not arduino_controller or not arduino_controller.connected:
        emit('log_message', {'message': 'Error: Arduino not connected', 'type': 'error'})
        return
    
    episodes = data.get('episodes', 20)
    duration = data.get('duration', 8.0)
    target = data.get('target', 180.0)
    
    training_active = True
    emit('log_message', {'message': f'Starting RL training: {episodes} episodes', 'type': 'command'})
    
    # Start training thread
    training_thread = threading.Thread(target=run_training, args=(episodes, duration, target), daemon=True)
    training_thread.start()

@socketio.on('stop_rl_training')
def handle_stop_training():
    global training_active
    training_active = False
    emit('log_message', {'message': 'Training stopped', 'type': 'warning'})

def start_status_monitoring():
    """Start status monitoring thread - FIXED to update visuals properly"""
    global status_monitoring_active
    
    status_monitoring_active = True
    print("🔍 Starting status monitoring thread...")
    
    def monitor():
        global status_monitoring_active
        consecutive_failures = 0
        max_failures = 5
        
        while status_monitoring_active and arduino_controller:
            try:
                if arduino_controller.connected:
                    # Get status from Arduino
                    status = arduino_controller.get_status()
                    
                    if status and len(status) > 1:  # Make sure we got real data
                        status['connected'] = True
                        
                        # Debug: Print what we're sending to web interface
                        print(f"📊 Status update: degrees={status.get('degrees', 'N/A')}, target={status.get('target_degrees', 'N/A')}, pid={status.get('pid_enabled', 'N/A')}")
                        
                        # Send to web interface
                        socketio.emit('arduino_status', status)
                        consecutive_failures = 0
                        
                    else:
                        consecutive_failures += 1
                        print(f"⚠️  Empty status response ({consecutive_failures}/{max_failures})")
                        
                        if consecutive_failures >= max_failures:
                            print("❌ Too many status failures, stopping monitoring")
                            status_monitoring_active = False
                            socketio.emit('arduino_status', {'connected': False})
                            break
                else:
                    print("❌ Arduino not connected, stopping monitoring")
                    socketio.emit('arduino_status', {'connected': False})
                    break
                    
            except Exception as e:
                consecutive_failures += 1
                print(f"❌ Status monitoring error ({consecutive_failures}/{max_failures}): {e}")
                
                if consecutive_failures >= max_failures:
                    status_monitoring_active = False
                    socketio.emit('arduino_status', {'connected': False})
                    break
            
            # Wait between status checks
            time.sleep(2.0)  # Slower updates to be more stable
        
        print("🔍 Status monitoring thread stopped")
        status_monitoring_active = False
    
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

def run_training(episodes, duration, target):
    """Run RL training"""
    global training_active, arduino_controller
    
    best_reward = -float('inf')
    best_params = None
    
    for episode in range(episodes):
        if not training_active:
            break
        
        # Generate random parameters
        kp = np.random.uniform(1, 20)
        ki = np.random.uniform(0, 2)
        kd = np.random.uniform(5, 50)
        
        # Test parameters
        reward = test_parameters(kp, ki, kd, target, duration)
        
        if reward > best_reward:
            best_reward = reward
            best_params = {'kp': kp, 'ki': ki, 'kd': kd}
        
        # Update training display
        socketio.emit('training_update', {
            'episode': episode + 1,
            'best_reward': best_reward,
            'best_params': best_params
        })
        
        time.sleep(1)
    
    training_active = False
    socketio.emit('log_message', {
        'message': f'Training complete! Best: Kp={best_params["kp"]:.2f}, Ki={best_params["ki"]:.2f}, Kd={best_params["kd"]:.2f}',
        'type': 'response'
    })

def test_parameters(kp, ki, kd, target, duration):
    """Test PID parameters and return reward"""
    try:
        # Reset system
        arduino_controller.send_command('PID_OFF')
        time.sleep(0.2)
        arduino_controller.send_command('ZERO')
        time.sleep(0.5)
        
        # Set parameters
        arduino_controller.send_command(f'PID {kp} {ki} {kd}')
        arduino_controller.send_command(f'TARGET {target}')
        arduino_controller.send_command('PID_ON')
        
        # Collect data
        start_time = time.time()
        errors = []
        
        while time.time() - start_time < duration:
            status = arduino_controller.get_status()
            error = abs(status.get('error', 0))
            errors.append(error)
            time.sleep(0.1)
        
        arduino_controller.send_command('PID_OFF')
        
        # Calculate reward
        if errors:
            mean_error = np.mean(errors)
            reward = max(0, 100 - mean_error)
        else:
            reward = 0
        
        return reward
        
    except Exception as e:
        logger.error(f"Parameter test error: {e}")
        return 0

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 PID CONTROLLER WEB SERVER (RESTORED)")
    print("=" * 60)
    
    local_ip = get_local_ip()
    
    print("Creating dashboard template...")
    create_dashboard_template()
    print("Dashboard template created!")
    print()
    print(f"📡 Starting server...")
    print(f"   Local:    http://localhost:5000")
    print(f"   Network:  http://{local_ip}:5000")
    print()
    print("🔧 Your original system restored with fixes:")
    print("   • PID motor control")
    print("   • Real-time position tracking") 
    print("   • Manual motor controls")
    print("   • Reinforcement learning training")
    print("   • Connection stability improved")
    print()
    print("⚠️  Required: pip install flask flask-socketio numpy")
    print("=" * 60)
    
    try:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        if arduino_controller:
            arduino_controller.disconnect()