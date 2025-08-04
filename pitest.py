#!/usr/bin/env python3
"""
Basic Python server to test Arduino connection
Just tests basic HTTP communication
"""

from flask import Flask, render_template_string
import requests
import socket
import time

app = Flask(__name__)

# Basic HTML template
TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Basic Arduino Test</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
        .status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
        button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; margin: 5px; cursor: pointer; }
        button:hover { background: #0056b3; }
        input { padding: 8px; margin: 5px; border: 1px solid #ccc; border-radius: 3px; }
        #log { background: #000; color: #0f0; padding: 15px; border-radius: 5px; height: 300px; overflow-y: scroll; font-family: monospace; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔧 Basic Arduino Connection Test</h1>
        
        <div class="status info">
            <strong>Python Server Info:</strong><br>
            Server running on: <a href="http://{{ server_ip }}:5000" target="_blank">http://{{ server_ip }}:5000</a><br>
            Local access: <a href="http://localhost:5000" target="_blank">http://localhost:5000</a>
        </div>
        
        <h2>Arduino Connection Test</h2>
        <div>
            <label>Arduino IP Address:</label>
            <input type="text" id="arduinoIP" value="192.168.0.224" placeholder="192.168.x.x">
            <button onclick="testConnection()">Test Connection</button>
            <button onclick="clearLog()">Clear Log</button>
        </div>
        
        <h3>Connection Log:</h3>
        <div id="log">Ready to test Arduino connection...\n</div>
    </div>

    <script>
        function log(message, type = 'info') {
            const logDiv = document.getElementById('log');
            const timestamp = new Date().toLocaleTimeString();
            const colors = { info: '#0f0', error: '#f00', success: '#0f0', warning: '#ff0' };
            const color = colors[type] || '#0f0';
            
            logDiv.innerHTML += `<span style="color: ${color}">[${timestamp}] ${message}</span>\n`;
            logDiv.scrollTop = logDiv.scrollHeight;
        }
        
        async function testConnection() {
            const ip = document.getElementById('arduinoIP').value;
            
            if (!ip) {
                log('Please enter Arduino IP address', 'error');
                return;
            }
            
            log(`Testing connection to Arduino at ${ip}...`, 'info');
            
            try {
                const response = await fetch('/test_arduino', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ip: ip })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    log(`✅ SUCCESS: ${result.message}`, 'success');
                    if (result.arduino_response) {
                        log(`Arduino says: ${result.arduino_response}`, 'info');
                    }
                } else {
                    log(`❌ FAILED: ${result.message}`, 'error');
                }
                
            } catch (error) {
                log(`❌ ERROR: ${error.message}`, 'error');
            }
        }
        
        function clearLog() {
            document.getElementById('log').innerHTML = 'Log cleared...\n';
        }
        
        // Auto-test on page load
        window.onload = function() {
            log('Page loaded. Ready to test!', 'info');
        };
    </script>
</body>
</html>
'''

def get_local_ip():
    """Get local IP address"""
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

@app.route('/')
def index():
    server_ip = get_local_ip()
    return render_template_string(TEMPLATE, server_ip=server_ip)

@app.route('/test_arduino', methods=['POST'])
def test_arduino():
    try:
        import json
        data = json.loads(request.get_data())
        arduino_ip = data.get('ip', '')
        
        if not arduino_ip:
            return json.dumps({'success': False, 'message': 'No IP provided'})
        
        print(f"Testing connection to Arduino at {arduino_ip}")
        
        # Test basic HTTP connection
        url = f"http://{arduino_ip}"
        
        try:
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                # Extract some info from response
                response_text = response.text[:200]  # First 200 chars
                
                return json.dumps({
                    'success': True, 
                    'message': f'Connected successfully to {arduino_ip}',
                    'arduino_response': f'HTTP {response.status_code} - Response received',
                    'preview': response_text
                })
            else:
                return json.dumps({
                    'success': False, 
                    'message': f'Arduino responded with HTTP {response.status_code}'
                })
                
        except requests.exceptions.ConnectTimeout:
            return json.dumps({
                'success': False, 
                'message': f'Connection timeout - Arduino not responding at {arduino_ip}'
            })
        except requests.exceptions.ConnectionError:
            return json.dumps({
                'success': False, 
                'message': f'Connection refused - Check Arduino IP and network'
            })
        except Exception as e:
            return json.dumps({
                'success': False, 
                'message': f'Connection error: {str(e)}'
            })
            
    except Exception as e:
        return json.dumps({'success': False, 'message': f'Server error: {str(e)}'})

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 BASIC ARDUINO CONNECTION TEST SERVER")
    print("=" * 60)
    print()
    
    local_ip = get_local_ip()
    
    print(f"📡 Server starting...")
    print(f"   Local:    http://localhost:5000")
    print(f"   Network:  http://{local_ip}:5000")
    print()
    print("📋 What this tests:")
    print("   • Python server can start")
    print("   • Basic HTTP connection to Arduino")
    print("   • Network connectivity")
    print("   • Response from Arduino")
    print()
    print("🔧 Instructions:")
    print("   1. Flash the basic Arduino code")
    print("   2. Open the web page above")
    print("   3. Enter Arduino IP and click 'Test Connection'")
    print()
    print("⚠️  Requirements: pip install flask requests")
    print("=" * 60)
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")