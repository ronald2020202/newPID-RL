#!/usr/bin/env python3
"""
Debug script to understand the connection issue
This will help us see exactly what's happening
"""

import socket
import time
import threading

def test_arduino_connection_detailed(arduino_ip, port=80):
    """Test connection with detailed logging"""
    print(f"\n🔍 DETAILED CONNECTION TEST TO {arduino_ip}:{port}")
    print("=" * 60)
    
    try:
        print("1. Creating socket...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        
        print("2. Attempting to connect...")
        start_time = time.time()
        sock.connect((arduino_ip, port))
        connect_time = time.time() - start_time
        print(f"   ✅ Connected in {connect_time:.2f} seconds")
        
        print("3. Waiting for initial response...")
        try:
            sock.settimeout(3.0)
            data = sock.recv(1024)
            initial_response = data.decode('utf-8', errors='ignore').strip()
            print(f"   📨 Initial response: '{initial_response}'")
        except socket.timeout:
            print("   ⚠️  No initial response (timeout)")
            initial_response = None
        
        print("4. Sending STATUS command...")
        sock.send(b"STATUS\n")
        
        try:
            data = sock.recv(1024)
            status_response = data.decode('utf-8', errors='ignore').strip()
            print(f"   📨 Status response: '{status_response[:100]}...'")
        except socket.timeout:
            print("   ❌ No status response (timeout)")
            status_response = None
        
        print("5. Testing connection persistence...")
        for i in range(3):
            print(f"   Test {i+1}/3: Sending PING...")
            try:
                sock.send(b"STATUS\n")
                data = sock.recv(1024)
                response = data.decode('utf-8', errors='ignore').strip()
                print(f"      📨 Response: OK ({len(response)} chars)")
                time.sleep(1)
            except Exception as e:
                print(f"      ❌ Failed: {e}")
                break
        
        print("6. Closing connection...")
        sock.close()
        print("   ✅ Connection closed normally")
        
        return True
        
    except socket.timeout:
        print("   ❌ CONNECTION TIMEOUT")
        return False
    except ConnectionRefused:
        print("   ❌ CONNECTION REFUSED - Arduino not listening")
        return False
    except Exception as e:
        print(f"   ❌ CONNECTION ERROR: {e}")
        return False

def monitor_arduino_behavior(arduino_ip, duration=30):
    """Monitor Arduino for connection patterns"""
    print(f"\n🔍 MONITORING ARDUINO BEHAVIOR FOR {duration} SECONDS")
    print("=" * 60)
    
    connection_count = 0
    start_time = time.time()
    
    while time.time() - start_time < duration:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((arduino_ip, 80))
            
            connection_count += 1
            timestamp = time.strftime("%H:%M:%S")
            
            # Try to get a response
            try:
                data = sock.recv(1024)
                response = data.decode('utf-8', errors='ignore').strip()
                print(f"[{timestamp}] Connection #{connection_count}: Response='{response[:50]}...'")
            except socket.timeout:
                print(f"[{timestamp}] Connection #{connection_count}: No response")
            
            sock.close()
            time.sleep(2)  # Wait 2 seconds between connections
            
        except Exception as e:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] Connection failed: {e}")
            time.sleep(2)
    
    print(f"\n📊 SUMMARY: {connection_count} successful connections in {duration} seconds")

def simulate_python_server_behavior(arduino_ip):
    """Simulate what the Python server is doing"""
    print(f"\n🔍 SIMULATING PYTHON SERVER BEHAVIOR")
    print("=" * 60)
    
    class MockArduinoController:
        def __init__(self, ip):
            self.arduino_ip = ip
            self.socket = None
            self.connected = False
        
        def connect(self):
            print("   🔌 Attempting connection...")
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(3.0)
                self.socket.connect((self.arduino_ip, 80))
                
                # Wait for response
                data = self.socket.recv(1024)
                response = data.decode('utf-8', errors='ignore').strip()
                print(f"   📨 Arduino says: '{response}'")
                
                self.connected = True
                return True
            except Exception as e:
                print(f"   ❌ Connection failed: {e}")
                self.connected = False
                return False
        
        def send_command(self, command):
            if not self.connected:
                return "ERROR Not connected"
            
            try:
                self.socket.send((command + '\n').encode())
                data = self.socket.recv(1024)
                return data.decode('utf-8', errors='ignore').strip()
            except Exception as e:
                print(f"   ❌ Command failed: {e}")
                self.connected = False
                return f"ERROR {e}"
        
        def get_status(self):
            response = self.send_command("STATUS")
            if response.startswith("STATUS "):
                return {"connected": True, "raw": response}
            return {"connected": False}
        
        def disconnect(self):
            print("   🔌 Disconnecting...")
            self.connected = False
            if self.socket:
                try:
                    self.socket.close()
                except:
                    pass

    # Test the mock controller
    controller = MockArduinoController(arduino_ip)
    
    print("1. Testing connection...")
    if controller.connect():
        print("2. Testing status command...")
        status = controller.get_status()
        print(f"   Status: {status}")
        
        print("3. Testing multiple status calls...")
        for i in range(3):
            status = controller.get_status()
            print(f"   Call {i+1}: {status.get('connected', False)}")
            time.sleep(0.5)
        
        print("4. Disconnecting...")
        controller.disconnect()
    else:
        print("   Connection failed - can't proceed with tests")

if __name__ == '__main__':
    arduino_ip = "192.168.0.224"  # Your Arduino IP
    
    print("🔧 ARDUINO CONNECTION ISSUE DEBUGGER")
    print("This will help us understand why connections keep disconnecting")
    print()
    
    # Test 1: Basic connection test
    test_arduino_connection_detailed(arduino_ip)
    
    # Test 2: Monitor behavior
    print("\n" + "="*60)
    response = input("Press ENTER to start monitoring Arduino behavior (30 seconds)...")
    monitor_arduino_behavior(arduino_ip, 30)
    
    # Test 3: Simulate Python server
    print("\n" + "="*60)
    response = input("Press ENTER to simulate Python server behavior...")
    simulate_python_server_behavior(arduino_ip)
    
    print("\n🎯 WHAT TO LOOK FOR:")
    print("• Does Arduino respond consistently?")
    print("• Are there timeouts or connection refusals?") 
    print("• Does the connection stay open or close immediately?")
    print("• Is Arduino handling multiple concurrent connections?")
    print("\nThis will help us identify if the issue is:")
    print("• Arduino closing connections too quickly")
    print("• Network/timing issues")
    print("• Python socket handling problems")
    print("• Status monitoring causing conflicts")